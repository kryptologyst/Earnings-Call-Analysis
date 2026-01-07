"""Seeding utilities for reproducible results."""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    torch.manual_seed(seed)
    
    # Set environment variables for additional reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        torch.device: CUDA, MPS (Apple Silicon), or CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
