#!/bin/bash

# Define the directory
DIR="../../kernels"

# Find and delete all .bin files in the directory
find "$DIR" -type f -name "*.bin" -exec rm -f {} \;

echo "All .bin files have been deleted from $DIR."
