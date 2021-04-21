'''
the script is to return the most similar poet from the input words
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf
import os
import argparse
import torch 


parser = argparse.ArgumentParser(description='fill arguement')

parser.add_argument('--model_name', type=str, required=False,
                    help='pretrain\'s model name or model path',
                    default='bert-base-chinese')