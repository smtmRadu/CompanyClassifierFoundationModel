import polars as pl
from ollama import chat
import torch._dynamo.config
from transformers import AutoTokenizer, ModernBertModel
import time
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import polars as pl
import numpy as np


