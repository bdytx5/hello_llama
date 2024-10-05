# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    

# model_args = ModelArgs(
#     dim=2048,                  # Hidden dimension size
#     n_layers=16,               # Number of layers
#     n_heads=32,                # Number of attention heads
#     n_kv_heads=8,              # Number of key-value heads
#     vocab_size=128256,         # Vocabulary size
#     multiple_of=256,           # Multiple for feed-forward dimension
#     ffn_dim_multiplier=1.5,    # Feed-forward dimension multiplier
#     norm_eps=1e-5,             # Normalization epsilon
#     rope_theta=500000.0,       # RoPE theta value
#     max_batch_size=32,         # Maximum batch size (arbitrary, can be adjusted)
#     max_seq_len=2048           # Maximum sequence length (default or can be adjusted)
# )



# import os
# import torch
# import torch.distributed as dist
# from fairscale.nn.model_parallel import initialize
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Set the required environment variables for distributed training
# os.environ['MASTER_ADDR'] = '127.0.0.1'  # IP address of the master node (localhost for single machine)
# os.environ['MASTER_PORT'] = '29500'      # A free port on the master node
# os.environ['RANK'] = '0'                 # Rank of the current process (0 for single process)
# os.environ['WORLD_SIZE'] = '1'           # Total number of processes (1 for single process)

# # Initialize the distributed process group
# dist.init_process_group(backend='nccl')  # Use 'gloo' for CPU or 'nccl' for GPU

# # Initialize model parallelism with the desired model_parallel_size
# initialize.initialize_model_parallel(model_parallel_size_=1)

# # Load the state dictionary from the checkpoint file
# checkpoint_path = "meta-llama/Llama-3.2-1B/original/consolidated.00.pth"
# state_dict = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU or 'cuda' if you have a GPU
# model = Transformer(model_args).to('cuda:0')
# # Load the weights into the instantiated model
# model.load_state_dict(state_dict, strict=False)


# # Set the model to evaluation mode
# model.eval()

# # Load the Hugging Face tokenizer
# tokenizer_path = "meta-llama/Llama-3.2-1B/original"  # Path to the directory containing the tokenizer files
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# # Encode the input text using the tokenizer
# input_text = "Once upon a time"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Convert to tensor

# # Ensure the input tensor is on the same device as the model
# input_ids = input_ids.to('cuda' if torch.cuda.is_available() else 'cpu')


# # Generate text using a simple greedy decoding approach
# def generate_text(model, input_ids, max_length=100):
#     # Initialize the generated sequence with the input ids
#     generated_ids = input_ids

#     # Generate tokens one-by-one
#     for _ in range(max_length):
#         # Forward pass through the model to get the logits
#         with torch.no_grad():
#             outputs = model(generated_ids, start_pos=0)  # Modify start_pos as needed

#         # Get the logits for the last token in the sequence
#         next_token_logits = outputs[:, -1, :]

#         # Select the next token using greedy sampling (select the token with the highest probability)
#         next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

#         # Append the new token to the generated sequence
#         generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

#         # Stop if EOS token is generated
#         if next_token_id.item() == tokenizer.eos_token_id:
#             break

#     return generated_ids

# # Generate text based on the input prompt
# generated_ids = generate_text(model, input_ids, max_length=100)

# # Decode the generated ids back to text
# generated_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)

# # Print the generated text
# print(f"Input: {input_text}")
# print(f"Generated Output: {generated_text}")

# # Destroy process group to ensure clean exit
# dist.destroy_process_group()


# import os
# import torch
# import torch.distributed as dist
# from torch.optim import Adam
# import torch.nn.functional as F
# from fairscale.nn.model_parallel import initialize
# from transformers import AutoTokenizer

# # Define the model configuration parameters
# model_args = ModelArgs(
#     dim=2048,                  # Hidden dimension size
#     n_layers=16,               # Number of layers
#     n_heads=32,                # Number of attention heads
#     n_kv_heads=8,              # Number of key-value heads
#     vocab_size=128256,         # Vocabulary size
#     multiple_of=256,           # Multiple for feed-forward dimension
#     ffn_dim_multiplier=1.5,    # Feed-forward dimension multiplier
#     norm_eps=1e-5,             # Normalization epsilon
#     rope_theta=500000.0,       # RoPE theta value
#     max_batch_size=32,         # Maximum batch size (arbitrary, can be adjusted)
#     max_seq_len=2048           # Maximum sequence length (default or can be adjusted)
# )

# # Set the required environment variables for distributed training
# os.environ['MASTER_ADDR'] = '127.0.0.1'  # IP address of the master node (localhost for single machine)
# os.environ['MASTER_PORT'] = '29500'      # A free port on the master node
# os.environ['RANK'] = '0'                 # Rank of the current process (0 for single process)
# os.environ['WORLD_SIZE'] = '1'           # Total number of processes (1 for single process)

# # Initialize the distributed process group
# if not dist.is_initialized():
#     dist.init_process_group(backend='nccl')  # Use 'gloo' for CPU or 'nccl' for GPU

# # Initialize model parallelism with the desired model_parallel_size
# initialize.initialize_model_parallel(model_parallel_size_=1)

# # Load the state dictionary from the checkpoint file
# checkpoint_path = "meta-llama/Llama-3.2-1B/original/consolidated.00.pth"
# state_dict = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU or 'cuda' if you have a GPU

# # Instantiate the Transformer model with the provided configuration and move it to the appropriate device
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Transformer(model_args).to(device)

# # Load the weights into the instantiated model
# model.load_state_dict(state_dict, strict=False)

# # Set the model to training mode (to enable gradient calculation)
# model.train()

# # Load the Hugging Face tokenizer
# tokenizer_path = "meta-llama/Llama-3.2-1B/original"  # Path to the directory containing the tokenizer files
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# # Encode the input text using the tokenizer
# input_text = "Once upon a time"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Convert to tensor

# # Ensure the input tensor is on the same device as the model
# input_ids = input_ids.to(device)

# # Create target tensor by shifting input_ids (in this simple example, we use input_ids as both input and target)
# target_ids = input_ids.clone()

# # Define an optimizer (e.g., Adam) for updating the model's parameters
# optimizer = Adam(model.parameters(), lr=1e-4)

# # Zero gradients for safety
# optimizer.zero_grad()

# # Forward pass through the model to get the logits (remove torch.no_grad() for gradient calculation)
# outputs = model(input_ids, start_pos=0)  # start_pos set to 0

# # Calculate loss using Cross Entropy Loss between the predicted logits and the target tokens
# # Note: `outputs` shape is (batch_size, sequence_length, vocab_size) and `target_ids` shape is (batch_size, sequence_length)
# loss = F.cross_entropy(outputs.view(-1, model_args.vocab_size), target_ids.view(-1))
# loss.requires_grad = True

# # Perform backward pass to compute gradients
# loss.backward()

# # Step the optimizer to update the model parameters
# optimizer.step()

# # Print the loss value
# print(f"Loss: {loss.item()}")

# # Set the model back to evaluation mode
# model.eval()

# # Generate text using a simple greedy decoding approach (same as previous implementation)
# def generate_text(model, input_ids, max_length=100):
#     # Initialize the generated sequence with the input ids
#     generated_ids = input_ids

#     # Generate tokens one-by-one
#     for _ in range(max_length):
#         # Forward pass through the model to get the logits
#         with torch.no_grad():  # Disable gradients for inference
#             outputs = model(generated_ids, start_pos=0)

#         # Get the logits for the last token in the sequence
#         next_token_logits = outputs[:, -1, :]

#         # Select the next token using greedy sampling (select the token with the highest probability)
#         next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

#         # Append the new token to the generated sequence
#         generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

#         # Stop if EOS token is generated
#         if next_token_id.item() == tokenizer.eos_token_id:
#             break

#     return generated_ids

# # Generate text based on the input prompt after training
# generated_ids = generate_text(model, input_ids, max_length=100)

# # Decode the generated ids back to text
# generated_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)

# # Print the generated text
# print(f"Input: {input_text}")
# print(f"Generated Output: {generated_text}")

# # Destroy process group to ensure clean exit
# dist.destroy_process_group()


# import os
# import torch
# import torch.distributed as dist
# from torch.optim import Adam
# import torch.nn.functional as F
# from fairscale.nn.model_parallel import initialize
# from transformers import AutoTokenizer

# # Define the model configuration parameters
# model_args = ModelArgs(
#     dim=2048,                  # Hidden dimension size
#     n_layers=16,               # Number of layers
#     n_heads=32,                # Number of attention heads
#     n_kv_heads=8,              # Number of key-value heads
#     vocab_size=128256,         # Vocabulary size
#     multiple_of=256,           # Multiple for feed-forward dimension
#     ffn_dim_multiplier=1.5,    # Feed-forward dimension multiplier
#     norm_eps=1e-5,             # Normalization epsilon
#     rope_theta=500000.0,       # RoPE theta value
#     max_batch_size=32,         # Maximum batch size (arbitrary, can be adjusted)
#     max_seq_len=2048           # Maximum sequence length (default or can be adjusted)
# )

# # Set the required environment variables for distributed training
# os.environ['MASTER_ADDR'] = '127.0.0.1'  # IP address of the master node (localhost for single machine)
# os.environ['MASTER_PORT'] = '29500'      # A free port on the master node
# os.environ['RANK'] = '0'                 # Rank of the current process (0 for single process)
# os.environ['WORLD_SIZE'] = '1'           # Total number of processes (1 for single process)

# # Initialize the distributed process group based on available device
# backend = 'nccl' if torch.cuda.is_available() else 'gloo'

# if not dist.is_initialized():
#     dist.init_process_group(backend=backend)  # Use 'nccl' for CUDA-enabled devices, 'gloo' for CPU

# # Initialize model parallelism with the desired model_parallel_size
# initialize.initialize_model_parallel(model_parallel_size_=1)

# # Dynamic device finding based on availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the state dictionary from the checkpoint file
# # Set the checkpoint and tokenizer paths
# checkpoint_path = os.path.expanduser("~/Desktop/weights/meta-llama/Llama-3.2-1B/original/consolidated.00.pth")
# state_dict = torch.load(checkpoint_path, map_location=device)  # Load to the appropriate device

# # Instantiate the Transformer model with the provided configuration and move it to the appropriate device
# model = Transformer(model_args).to(device)

# # Load the weights into the instantiated model
# model.load_state_dict(state_dict, strict=False)

# # Set the model to training mode (to enable gradient calculation)
# model.train()

# # Load the Hugging Face tokenizer
# tokenizer_path = os.path.expanduser("~/Desktop/weights/meta-llama/Llama-3.2-1B/original")  # Path to the directory containing the tokenizer files
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# # Encode the input text using the tokenizer
# input_text = "Once upon a time"
# input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Convert to tensor

# # Ensure the input tensor is on the same device as the model
# input_ids = input_ids.to(device)

# # Create target tensor by shifting input_ids (in this simple example, we use input_ids as both input and target)
# target_ids = input_ids.clone()

# # Training boolean flag to control if training is performed
# perform_training = True

# if perform_training:
#     # Define an optimizer (e.g., Adam) for updating the model's parameters
#     optimizer = Adam(model.parameters(), lr=1e-4)

#     # Zero gradients for safety
#     optimizer.zero_grad()

#     # Forward pass through the model to get the logits (remove torch.no_grad() for gradient calculation)
#     outputs = model(input_ids, start_pos=0)  # start_pos set to 0

#     # Calculate loss using Cross Entropy Loss between the predicted logits and the target tokens
#     # Note: `outputs` shape is (batch_size, sequence_length, vocab_size) and `target_ids` shape is (batch_size, sequence_length)
#     loss = F.cross_entropy(outputs.view(-1, model_args.vocab_size), target_ids.view(-1))
#     loss.requires_grad = True

#     # Perform backward pass to compute gradients
#     loss.backward()

#     # Step the optimizer to update the model parameters
#     optimizer.step()

#     # Print the loss value
#     print(f"Loss: {loss.item()}")

# # Set the model back to evaluation mode
# model.eval()

# # Generate text using a simple greedy decoding approach (same as previous implementation)
# def generate_text(model, input_ids, max_length=100):
#     # Initialize the generated sequence with the input ids
#     generated_ids = input_ids

#     # Generate tokens one-by-one
#     for _ in range(max_length):
#         # Forward pass through the model to get the logits
#         with torch.no_grad():  # Disable gradients for inference
#             outputs = model(generated_ids, start_pos=0)

#         # Get the logits for the last token in the sequence
#         next_token_logits = outputs[:, -1, :]

#         # Select the next token using greedy sampling (select the token with the highest probability)
#         next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

#         # Append the new token to the generated sequence
#         generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

#         # Stop if EOS token is generated
#         if next_token_id.item() == tokenizer.eos_token_id:
#             break

#     return generated_ids

# # Generate text based on the input prompt after training
# generated_ids = generate_text(model, input_ids, max_length=100)

# # Decode the generated ids back to text
# generated_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)

# # Print the generated text
# print(f"Input: {input_text}")
# print(f"Generated Output: {generated_text}")

# # Destroy process group to ensure clean exit
# dist.destroy_process_group()
