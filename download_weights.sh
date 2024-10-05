#!/bin/bash

# Create the directory if it doesn't exist
mkdir -p ~/Desktop/weights/meta-llama/Llama-3.2-1B

# Download the model using huggingface-cli with the specified sub-directory and parameters
huggingface-cli download meta-llama/Llama-3.2-1B --include "original/*" --local-dir ~/Desktop/weights/meta-llama/Llama-3.2-1B
