#!/usr/bin/env python
# coding: utf-8

"""
Utility functions for PRISM package.
"""

import torch


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
