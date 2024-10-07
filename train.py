import math
import random
import tqdm
import numpy as np
from torch import Tensor
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import os
import sentencepiece as spm
import glob
import re

from minGRU_pytorch.minGRULM import minGRULM

# **Import SummaryWriter for TensorBoard**
from torch.utils.tensorboard import SummaryWriter  # Added for TensorBoard logging

class ChunkedTextDataset(IterableDataset):
    def __init__(self, data_file, seq_len, chunk_size=1024 * 1024):
        self.data_file = data_file
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.sp = spm.SentencePieceProcessor()
        self.sp.load('tokenizer.model')  # Make sure tokenizer.model is in the correct location

    def __iter__(self):
        with open(self.data_file, 'r', encoding='utf-8') as file:  # Open in text mode
            while True:
                chunk = file.read(self.chunk_size)
                if not chunk:
                    break  # End of file

                token_ids = self.sp.encode(chunk)

                for i in range(0, len(token_ids) - self.seq_len - 1, self.seq_len):
                    yield torch.tensor(token_ids[i:i + self.seq_len + 1], dtype=torch.long)


def remove_module_prefix(state_dict):
    """
    Removes 'module.' prefix from state_dict keys if present.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[7:]  # Remove 'module.' prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def main(rank, world_size, train_data_file, val_data_file, args):
    try:
        # Initialize the process group only if world_size > 1
        if world_size > 1:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '5554'
            dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
            # Set the device for each process
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0  # Since there's only one process

        print(f"Rank {rank} using device: {device}")

        # Load validation data (only on rank 0)
        if rank == 0:
            val_data = np.load(val_data_file, allow_pickle=True)
            val_data = val_data.tolist()
        else:
            val_data = None  # Other ranks don't need validation data

        # Constants and Hyperparameters
        NUM_EPOCHS = args.num_epochs  # Number of epochs to train
        BATCH_SIZE = args.batch_size
        GRAD_ACCUM_EVERY = args.grad_accum_every
        LEARNING_RATE = args.learning_rate
        VALIDATE_EVERY = args.validate_every
        CHECKPOINT_EVERY = args.checkpoint_every  # Save checkpoint every n steps
        CHECKPOINT_PATH = args.checkpoint_path  # Path to load checkpoint (if any)
        PRIME_LENGTH = 64
        GENERATE_EVERY = args.generate_every
        GENERATE_LENGTH = 128
        SEQ_LEN = 1024
        SAVE_LAST = args.save_last  # Number of last checkpoints to keep

        # **Initialize TensorBoard SummaryWriter (only for rank 0)**
        if rank == 0:
            writer = SummaryWriter(log_dir=args.log_dir)  # Initialize SummaryWriter
            print(f"TensorBoard logging enabled. Logs will be saved to {args.log_dir}")
        else:
            writer = None  # Other ranks do not write to TensorBoard

        # Load the SentencePiece tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load('tokenizer.model')
        vocab_size = sp.get_piece_size()

        if rank == 0:
            print(f"Vocabulary size: {vocab_size}")

        # The minGRU language model
        model = minGRULM(
            num_tokens=vocab_size,
            dim=1024,
            depth=12
        ).to(device)


        # Create datasets
        train_dataset = ChunkedTextDataset(train_data_file, SEQ_LEN)

        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset)
        else:
            train_sampler = None

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                  shuffle=(train_sampler is None), num_workers=6, pin_memory=True)
        if rank == 0:  # Validation loader only on rank 0
            val_dataset = ChunkedTextDataset(val_data_file, SEQ_LEN)  # Use ChunkedTextDataset
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        # Optimizer
        optim = Adam(model.parameters(), lr=LEARNING_RATE)

        # Mixed Precision Training
        scaler = GradScaler()

        # Initialize training state
        start_epoch = 0
        start_step = 0

        # Checkpoint Loading
        if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
            if rank == 0:
                print(f"Loading checkpoint from {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            state_dict = checkpoint['model_state_dict']
            state_dict = remove_module_prefix(state_dict)  # Remove 'module.' prefix if present
            model.load_state_dict(state_dict)
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            if world_size > 1:
                # Wrap the model with DDP after loading state_dict
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
            if rank == 0:
                print(f"Resumed from epoch {start_epoch}, step {start_step}")
                
        else:
            # Wrap the model with DDP only if world_size > 1 and not loading a checkpoint
            if world_size > 1:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        # Sampling helpers (same as before)
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
            return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim=keepdim)

        def top_k(logits, thres=0.9):
            k = max(1, int((1 - thres) * logits.shape[-1]))
            val, ind = torch.topk(logits, k)
            probs = torch.full_like(logits, float('-inf'))
            probs.scatter_(-1, ind, val)
            return probs

        def base_decoding(
            net,
            prompt: Tensor,
            seq_len: int,
            temperature = 1.,
            filter_thres = 0.9,
        ):
            prompt_seq_len, out = prompt.shape[-1], prompt.clone()
            sample_num_times = max(0, seq_len - prompt_seq_len)

            prev_hiddens = None

            for _ in range(sample_num_times):
                logits, prev_hiddens = net(out, return_prev_hiddens = True)
                logits = logits[:, -1]

                logits = top_k(logits, thres = filter_thres)
                sample = gumbel_sample(logits, temperature = temperature, dim = -1)

                out = torch.cat((out, sample), dim = -1)

            return out[..., prompt_seq_len:]

        def decode_tokens(token_ids):
            return sp.decode(token_ids)

        # Training Loop
        for epoch in range(start_epoch, NUM_EPOCHS):
            if world_size > 1:
                train_sampler.set_epoch(epoch)  # For shuffling with DistributedSampler

            if rank == 0:
                # Calculate the number of steps in this epoch
                steps_in_epoch = len(train_loader)
                print(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}, which has {steps_in_epoch} steps.")

            for batch_idx, data in enumerate(train_loader):
                step = start_step + batch_idx
                model.train()
                optim.zero_grad()

                for _ in range(GRAD_ACCUM_EVERY):
                    data = data.to(device)

                    with autocast():
                        loss = model(data, return_loss=True)
                        loss = loss / GRAD_ACCUM_EVERY

                    scaler.scale(loss).backward()

                # Gradient clipping and optimizer step
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

                # **Log Training Loss to TensorBoard (only on rank 0)**
                if rank == 0 and writer is not None:
                    writer.add_scalar('Train/Loss', loss.item() * GRAD_ACCUM_EVERY, global_step=step)

                # Validation and Generation (only on rank 0)
                if rank == 0:
                    if step % VALIDATE_EVERY == 0:
                        model.eval()
                        with torch.no_grad():
                            try:
                                val_data_batch = next(iter(val_loader))
                            except StopIteration:
                                val_data_batch = None

                            if val_data_batch is not None:
                                data_val = val_data_batch.to(device)

                                with autocast():
                                    val_loss = model(data_val, return_loss=True)
                                print(f"Validation loss at epoch {epoch + 1}, step {step}: {val_loss.item():.3f}")

                                # **Log Validation Loss to TensorBoard**
                                if writer is not None:
                                    writer.add_scalar('Validation/Loss', val_loss.item(), global_step=step)
                        model.train()

                    if step % GENERATE_EVERY == 0:
                        model.eval()
                        with torch.no_grad():
                            # Sample a random starting point in validation data
                            start_idx = random.randint(0, len(val_data) - PRIME_LENGTH - 1)
                            inp = torch.tensor(val_data[start_idx:start_idx + PRIME_LENGTH], dtype=torch.long).to(device)

                            prime = decode_tokens(inp.tolist())
                            print(f"Prime text:\n{prime}\n{'*' * 100}")

                            # **Log Prime Text to TensorBoard**
                            if writer is not None:
                                writer.add_text('Generate/Prime', prime, global_step=step)

                            prompt = inp.unsqueeze(0)  # Add batch dimension

                            sampled = base_decoding(model, prompt, PRIME_LENGTH + GENERATE_LENGTH)
                            sampled_ids = sampled[0].tolist()[PRIME_LENGTH:]  # Exclude the prime tokens

                            base_decode_output = decode_tokens(sampled_ids)

                            print(f"Generated text:\n{base_decode_output}\n")

                            # **Log Generated Text to TensorBoard**
                            if writer is not None:
                                writer.add_text('Generate/Output', base_decode_output, global_step=step)
                            
                            
                        model.train()

                    # Checkpointing
                    if step % CHECKPOINT_EVERY == 0:
                        checkpoint = {
                            'epoch': epoch,
                            'step': step,
                            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                        }
                        checkpoint_filename = f'checkpoint-step-{step}.pt'
                        torch.save(checkpoint, checkpoint_filename)
                        print(f"Checkpoint saved at step {step} to {checkpoint_filename}")

                        # Implement rolling checkpoint: keep only the last SAVE_LAST checkpoints
                        if SAVE_LAST > 0:
                            checkpoint_pattern = 'checkpoint-step-*.pt'
                            checkpoint_files = glob.glob(checkpoint_pattern)

                            # Extract step numbers and sort the files by step number
                            def extract_step(filename):
                                match = re.match(r'checkpoint-step-(\d+)\.pt', filename)
                                if match:
                                    return int(match.group(1))
                                else:
                                    return -1  # Non-matching files are considered older

                            checkpoint_files = sorted(checkpoint_files, key=lambda x: extract_step(x))

                            # Keep only the last SAVE_LAST checkpoints
                            if len(checkpoint_files) > SAVE_LAST:
                                old_checkpoints = checkpoint_files[:-SAVE_LAST]
                                for old_cp in old_checkpoints:
                                    try:
                                        os.remove(old_cp)
                                        print(f"Removed old checkpoint: {old_cp}")
                                    except Exception as e:
                                        print(f"Error removing checkpoint {old_cp}: {e}")

                # Update the starting step after each epoch
                start_step += len(train_loader)

        # Final Checkpoint after training
        if rank == 0:
            checkpoint = {
                'epoch': NUM_EPOCHS,
                'step': start_step,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            checkpoint_filename = f'checkpoint-final.pt'
            torch.save(checkpoint, checkpoint_filename)
            print(f"Final checkpoint saved to {checkpoint_filename}")

            # **Close the TensorBoard writer**
            if writer is not None:
                writer.close()
                print("TensorBoard writer closed.")

        # Clean up
        if world_size > 1:
            dist.destroy_process_group()

    except Exception as e:
        print(f"Exception in process {rank}: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception if you want the process to terminate
        raise e


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train minGRULM with checkpointing, multiple epochs, and dynamic data files.')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
    parser.add_argument('--grad_accum_every', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--validate_every', type=int, default=200, help='Validate every n steps.')
    parser.add_argument('--generate_every', type=int, default=300, help='Generate text every n steps.')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every n steps.')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint to resume training.')

    # New arguments for data files
    parser.add_argument('--train_data', type=str, default='train_data.npy', help='Path to training data file.')
    parser.add_argument('--val_data', type=str, default='val_data.npy', help='Path to validation data file.')

    # New argument for rolling checkpoint
    parser.add_argument('--save_last', type=int, default=2, help='Number of last checkpoints to keep.')

    # **New argument for TensorBoard log directory**
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory to save TensorBoard logs.')

    args = parser.parse_args()

    # To run with multiprocessing
    world_size = torch.cuda.device_count()

    # **FOR KAGGLE ENVIRONMENT**, it's recommended to use world_size=1 to avoid socket issues.
    # Uncomment the following line to force single-process training:
    # world_size = 1

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5554'

        mp.spawn(main,
                 args=(world_size, args.train_data, args.val_data, args),
                 nprocs=world_size,
                 join=True)
    else:
        main(0, world_size, args.train_data, args.val_data, args)
