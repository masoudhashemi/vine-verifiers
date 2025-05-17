#!/usr/bin/env python
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

# Import and run the main module
from verifiers.examples.train_vineppo_new import main

if __name__ == "__main__":
    main() 