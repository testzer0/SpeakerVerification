"""
Adithya Bhaskar, 2022.
This file contains global variables that are not 
user-defined.
"""

import torch

def get_device():
    """
    Are we executing on a CPU or GPU?
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

device = get_device()

if __name__ == '__main__':
    pass