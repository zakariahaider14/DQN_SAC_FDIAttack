# !#Testing with WAC
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import CosineDecay
import csv
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import json
from tensorflow.keras.layers import Layer, Dense, LSTM


import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC, DQN

# from DiscreteHybridEnv import DiscreteHybridEnv
# from combined_pinn import CompetingHybridEnv


import sys
import os
import tempfile
import json

# Set a different temporary directory
os.environ['TMPDIR'] = tempfile.gettempdir()
os.environ['TORCH_HOME'] = tempfile.gettempdir()

# Disable PyTorch's JIT compilation
os.environ['PYTORCH_JIT'] = '0'

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
torch.set_num_threads(1)

from datetime import datetime  # Change this line


# import shimmy
# Check if TensorFlow can see the GPU
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("GPU Device Name: ", tf.test.gpu_device_name())
