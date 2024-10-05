
import os
import torch
import torch.distributed as dist
from torch.optim import Adam
import torch.nn.functional as F
from fairscale.nn.model_parallel import initialize
from transformers import AutoTokenizer
from llama import ModelArgs, Transformer


perform_training = False
# weights_dir = "/home/brett/Desktop/weights/meta-llama/Llama-3.2-1B/original/"

# Define the model configuration parameters
model_args = ModelArgs(
    dim=2048,                  # Hidden dimension size
    n_layers=16,               # Number of layers
    n_heads=32,                # Number of attention heads
    n_kv_heads=8,              # Number of key-value heads
    vocab_size=128256,         # Vocabulary size
    multiple_of=256,           # Multiple for feed-forward dimension
    ffn_dim_multiplier=1.5,    # Feed-forward dimension multiplier
    norm_eps=1e-5,             # Normalization epsilon
    rope_theta=500000.0,       # RoPE theta value
    max_batch_size=32,         # Maximum batch size (arbitrary, can be adjusted)
    max_seq_len=2048           # Maximum sequence length (default or can be adjusted)
)

# Set the required environment variables for distributed training
os.environ['MASTER_ADDR'] = '127.0.0.1'  # IP address of the master node (localhost for single machine)
os.environ['MASTER_PORT'] = '29500'      # A free port on the master node
os.environ['RANK'] = '0'                 # Rank of the current process (0 for single process)
os.environ['WORLD_SIZE'] = '1'           # Total number of processes (1 for single process)

# Initialize the distributed process group based on available device
backend = 'nccl' if torch.cuda.is_available() else 'gloo'

if not dist.is_initialized():
    dist.init_process_group(backend=backend)  # Use 'nccl' for CUDA-enabled devices, 'gloo' for CPU

# Initialize model parallelism with the desired model_parallel_size
initialize.initialize_model_parallel(model_parallel_size_=1)

# Dynamic device finding based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the state dictionary from the checkpoint file
# Set the checkpoint and tokenizer paths
checkpoint_path = os.path.expanduser("~/Desktop/weights/meta-llama/Llama-3.2-1B/original/consolidated.00.pth")
state_dict = torch.load(checkpoint_path, map_location='cpu')  # Load to the appropriate device

# Instantiate the Transformer model with the provided configuration and move it to the appropriate device
model = Transformer(model_args).to(device)

# Load the weights into the instantiated model
model.load_state_dict(state_dict, strict=False)

# Set the model to training mode (to enable gradient calculation)
model.train()

# Load the Hugging Face tokenizer
tokenizer_path = os.path.expanduser("~/Desktop/weights/meta-llama/Llama-3.2-1B/original")  # Path to the directory containing the tokenizer files
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# Encode the input text using the tokenizer
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')  # Convert to tensor

# Ensure the input tensor is on the same device as the model
input_ids = input_ids.to(device)

# Create target tensor by shifting input_ids (in this simple example, we use input_ids as both input and target)
target_ids = input_ids.clone()

# Training boolean flag to control if training is performed

if perform_training:
    # Define an optimizer (e.g., Adam) for updating the model's parameters
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Zero gradients for safety
    optimizer.zero_grad()

    # Forward pass through the model to get the logits (remove torch.no_grad() for gradient calculation)
    outputs = model(input_ids, start_pos=0)  # start_pos set to 0

    # Calculate loss using Cross Entropy Loss between the predicted logits and the target tokens
    # Note: `outputs` shape is (batch_size, sequence_length, vocab_size) and `target_ids` shape is (batch_size, sequence_length)
    loss = F.cross_entropy(outputs.view(-1, model_args.vocab_size), target_ids.view(-1))
    loss.requires_grad = True

    # Perform backward pass to compute gradients
    loss.backward()

    # Step the optimizer to update the model parameters
    optimizer.step()

    # Print the loss value
    print(f"Loss: {loss.item()}")

# Set the model back to evaluation mode
model.eval()

# Generate text using a simple greedy decoding approach (same as previous implementation)
def generate_text(model, input_ids, max_length=100):
    # Initialize the generated sequence with the input ids
    generated_ids = input_ids

    # Generate tokens one-by-one
    for _ in range(max_length):
        # Forward pass through the model to get the logits
        with torch.no_grad():  # Disable gradients for inference
            outputs = model(generated_ids, start_pos=0)

        # Get the logits for the last token in the sequence
        next_token_logits = outputs[:, -1, :]

        # Select the next token using greedy sampling (select the token with the highest probability)
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # Append the new token to the generated sequence
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return generated_ids

# Generate text based on the input prompt after training
generated_ids = generate_text(model, input_ids, max_length=100)

# Decode the generated ids back to text
generated_text = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)

# Print the generated text
print(f"Input: {input_text}")
print(f"Generated Output: {generated_text}")

# Destroy process group to ensure clean exit
dist.destroy_process_group()
