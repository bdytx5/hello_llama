#!/bin/bash

# Define the model name and local directory paths
MODEL_NAME="meta-llama/Llama-3.2-1B"
LOCAL_DIR="$HOME/Desktop/weights"

# Create the directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Download the model using huggingface-cli
huggingface-cli download $MODEL_NAME --include "original/*" --local-dir "$LOCAL_DIR"
