#!/bin/bash

# This should be run first before any other scripts
# Load Sherlock modules needed for the program
ml python/3.12.1
ml cuda/12.8.0
ml ffmpeg/7.1

# Export Hugging Face cache directory
export HF_HOME=$SCRATCH/mirora/.cache
