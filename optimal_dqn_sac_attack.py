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

from DiscreteHybridEnv import DiscreteHybridEnv
from combined_pinn import CompetingHybridEnv


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

# ... existing code ...
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 

log_file = f"training_log_{current_time}.txt"

# Create a custom logger class
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to both terminal and file
sys.stdout = Logger(log_file)


# import shimmy
# Check if TensorFlow can see the GPU
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("GPU Device Name: ", tf.test.gpu_device_name())

NUM_BUSES = 33
NUM_EVCS = 5
EVCS_BUSES = [4, 10, 15, 20, 25]  # 0-based indexing

# Base Values
S_BASE = 10e6      # VA
V_BASE_HV = 12660  # V
V_BASE_LV = 800    # V
V_BASE_DC = 800    # V

# Calculate base currents and impedances
I_BASE_HV = S_BASE / (np.sqrt(3) * V_BASE_HV)
I_BASE_LV = S_BASE / (np.sqrt(3) * V_BASE_LV)
I_BASE_DC = S_BASE / V_BASE_DC
Z_BASE_HV = V_BASE_HV**2 / S_BASE
Z_BASE_LV = V_BASE_LV**2 / S_BASE

# EVCS Parameters
EVCS_CAPACITY = 80e3 / S_BASE  # 80 kW in per-unit
EVCS_EFFICIENCY = 0.98
EVCS_VOLTAGE = V_BASE_DC / V_BASE_LV  # In p.u.

GRID_VOLTAGE = 12600  # 12.6 kV

V_OUT_NOMINAL = EVCS_VOLTAGE  # Nominal output voltage in p.u.
V_OUT_VARIATION = 0.05  # 5% allowed variation

# Controller Parameters
EVCS_PLL_KP = 0.1
EVCS_PLL_KI = 0.2
MAX_PLL_ERROR = 5.0

EVCS_OUTER_KP = 0.5
EVCS_OUTER_KI = 0.3

EVCS_INNER_KP = 0.5
EVCS_INNER_KI = 0.3
OMEGA_N = 2 * np.pi * 60  # Nominal angular frequency (60 Hz)

# Wide Area Controller Parameters
WAC_KP_VDC = 0.3
WAC_KI_VDC = 0.2

WAC_KP_VOUT = 0.3
WAC_KI_VOUT = 0.2
WAC_VOLTAGE_SETPOINT = V_BASE_DC / V_BASE_LV  # Desired DC voltage in p.u.
WAC_VOUT_SETPOINT = V_BASE_DC / V_BASE_LV  # Desired output voltage in p.u.

CONTROL_MAX = 1.0
CONTROL_MIN = -1.0
INTEGRAL_MAX = 10.0
INTEGRAL_MIN = -10.0

# Other Parameters
CONSTRAINT_WEIGHT = 1.0
LCL_L1 = 30e-6 / Z_BASE_LV  # Convert to p.u.
LCL_L2 = 55e-6 / Z_BASE_LV  # Convert to p.u.
LCL_CF = 10e-6 * S_BASE / (V_BASE_LV**2)  # Convert to p.u.
R = 0.01 / Z_BASE_LV  # Convert to p.u.
C_dc = 100e-6 * S_BASE / (V_BASE_LV**2)  # Convert to p.u.

L_dc = 30e-6 / Z_BASE_LV  # Convert to p.u.
v_battery = 800 / V_BASE_DC  # Convert to p.u.
R_battery = 0.01 / Z_BASE_LV  # Convert to p.u.

# Time parameters
TIME_STEP = 1e-3  # 1 ms in seconds
TOTAL_TIME = 100  # 100 seconds

# Load IEEE 33-bus system data
line_data = [
    (1, 2, 0.0922, 0.0477), (2, 3, 0.493, 0.2511), (3, 4, 0.366, 0.1864), (4, 5, 0.3811, 0.1941),
    (5, 6, 0.819, 0.707), (6, 7, 0.1872, 0.6188), (7, 8, 1.7114, 1.2351), (8, 9, 1.03, 0.74),
    (9, 10, 1.04, 0.74), (10, 11, 0.1966, 0.065), (11, 12, 0.3744, 0.1238), (12, 13, 1.468, 1.155),
    (13, 14, 0.5416, 0.7129), (14, 15, 0.591, 0.526), (15, 16, 0.7463, 0.545), (16, 17, 1.289, 1.721),
    (17, 18, 0.732, 0.574), (2, 19, 0.164, 0.1565), (19, 20, 1.5042, 1.3554), (20, 21, 0.4095, 0.4784),
    (21, 22, 0.7089, 0.9373), (3, 23, 0.4512, 0.3083), (23, 24, 0.898, 0.7091), (24, 25, 0.896, 0.7011),
    (6, 26, 0.203, 0.1034), (26, 27, 0.2842, 0.1447), (27, 28, 1.059, 0.9337), (28, 29, 0.8042, 0.7006),
    (29, 30, 0.5075, 0.2585), (30, 31, 0.9744, 0.963), (31, 32, 0.31, 0.3619), (32, 33, 0.341, 0.5302)
]

bus_data = np.array([
    [1, 0, 0, 0], [2, 100, 60, 0], [3, 70, 40, 0], [4, 120, 80, 0], [5, 80, 30, 0],
    [6, 60, 20, 0], [7, 145, 100, 0], [8, 160, 100, 0], [9, 60, 20, 0], [10, 60, 20, 0],
    [11, 100, 30, 0], [12, 60, 35, 0], [13, 60, 35, 0], [14, 80, 80, 0], [15, 100, 10, 0],
    [16, 100, 20, 0], [17, 60, 20, 0], [18, 90, 40, 0], [19, 90, 40, 0], [20, 90, 40, 0],
    [21, 90, 40, 0], [22, 90, 40, 0], [23, 90, 40, 0], [24, 420, 200, 0], [25, 380, 200, 0],
    [26, 100, 25, 0], [27, 60, 25, 0], [28, 60, 20, 0], [29, 120, 70, 0], [30, 200, 600, 0],
    [31, 150, 70, 0], [32, 210, 100, 0], [33, 60, 40, 0]
])

# Convert bus data to per-unit
bus_data[:, 1:3] = bus_data[:, 1:3] * 1e3 / S_BASE

# Initialize Y-bus matrix
Y_bus = np.zeros((NUM_BUSES, NUM_BUSES), dtype=complex)



# Fill Y-bus matrix
for line in line_data:
    from_bus, to_bus, r, x = line
    from_bus, to_bus = int(from_bus)-1 , int(to_bus)-1 # Convert to 0-based index
    y = 1/complex(r, x)
    Y_bus[from_bus, from_bus] += y
    Y_bus[to_bus, to_bus] += y
    Y_bus[from_bus, to_bus] -= y
    Y_bus[to_bus, from_bus] -= y

# Convert to TensorFlow constant
Y_bus_tf = tf.constant(Y_bus, dtype=tf.complex64)


G_d = None
G_q = None

def initialize_conductance_matrices():
    """Initialize conductance matrices from Y-bus matrix"""
    global G_d, G_q, B_d, B_q
    # Extract G (conductance) and B (susceptance) matrices
    G_d = tf.cast(tf.math.real(Y_bus_tf), dtype=tf.float32)  # Real part for d-axis
    G_q = tf.cast(tf.math.real(Y_bus_tf), dtype=tf.float32)  # Real part for q-axis
    B_d = tf.cast(tf.math.imag(Y_bus_tf), dtype=tf.float32)  # Imaginary part for d-axis
    B_q = tf.cast(tf.math.imag(Y_bus_tf), dtype=tf.float32)  # Imaginary part for q-axis
    return G_d, G_q, B_d, B_q

# Call this function before training starts
G_d, G_q, B_d, B_q = initialize_conductance_matrices()

# For individual elements (if needed)
G_d_kh = tf.linalg.diag_part(G_d)  # Diagonal elements for d-axis conductance
G_q_kh = tf.linalg.diag_part(G_q)  # Diagonal elements for q-axis conductance
B_d_kh = tf.linalg.diag_part(B_d)  # Diagonal elements for d-axis susceptance
B_q_kh = tf.linalg.diag_part(B_q)  # Diagonal elements for q-axis susceptance



class SACWrapper(gym.Env):
    def __init__(self, env, agent_type, dqn_agent=None, sac_defender=None, sac_attacker=None):
        super(SACWrapper, self).__init__()
        
        self.env = env
        self.agent_type = agent_type
        self.dqn_agent = dqn_agent
        self.sac_defender = sac_defender
        self.sac_attacker = sac_attacker
        self.NUM_EVCS = env.NUM_EVCS
        
        # Initialize tracking variables as instance variables (not local variables)
        self.voltage_deviations = np.zeros(self.NUM_EVCS, dtype=np.float32)
        self.cumulative_deviation = 0.0
        self.attack_active = False
        self.target_evcs = np.zeros(self.NUM_EVCS)
        self.attack_duration = 0.0
        self.state = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        
        # Set action spaces
        self.observation_space = env.observation_space
        if agent_type == 'attacker':
            self.action_space = env.sac_attacker_action_space
        else:
            self.action_space = env.sac_defender_action_space

    def decode_dqn_action(self, action):
        """Decode DQN action by delegating to the underlying environment."""
        if hasattr(self.env, 'decode_dqn_action'):
            return self.env.decode_dqn_action(action)
        elif hasattr(self.env, 'decode_action'):  # Fallback to decode_action if available
            return self.env.decode_action(action)
        else:
            raise AttributeError("Neither decode_dqn_action nor decode_action method found in environment")

    def step(self, action):
        try:
            # Store current state of tracking variables in case of error
            current_tracking = {
                'voltage_deviations': self.voltage_deviations.copy(),
                'cumulative_deviation': self.cumulative_deviation,
                'attack_active': self.attack_active,
                'target_evcs': self.target_evcs.copy(),
                'attack_duration': self.attack_duration
            }

            # Convert input action to numpy array
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            
            # Get DQN action
            dqn_state = np.asarray(self.state).reshape(1, -1)
            dqn_raw = self.dqn_agent.predict(dqn_state, deterministic=True)
            dqn_action = np.asarray(dqn_raw[0] if isinstance(dqn_raw, tuple) else dqn_raw, dtype=np.int32)
            
            # Process DQN action using wrapper's decode method
            dqn_action = self.decode_dqn_action(dqn_action)  # Updated to use wrapper's method
            
            # Get agent actions
            if self.agent_type == 'attacker':
                attacker_action = action
                defender_action = (
                    self.sac_defender.predict(self.state, deterministic=True)[0] 
                    if self.sac_defender is not None 
                    else np.zeros(self.NUM_EVCS * 2, dtype=np.float32)
                )
            else:
                defender_action = action
                attacker_action = (
                    self.sac_attacker.predict(self.state, deterministic=True)[0] 
                    if self.sac_attacker is not None 
                    else np.zeros(self.NUM_EVCS * 2, dtype=np.float32)
                )
            
            # Combine actions
            combined_action = {
                'dqn': dqn_action,
                'attacker': np.asarray(attacker_action, dtype=np.float32),
                'defender': np.asarray(defender_action, dtype=np.float32)
            }
            
            # Take step
            next_state, rewards, done, truncated, info = self.env.step(combined_action)
            
            # Update state and tracking variables
            self.state = np.asarray(next_state, dtype=np.float32)
            info_dict = info if isinstance(info, dict) else {}
            
            # Update instance variables with new values
            self.voltage_deviations = np.asarray(info_dict.get('voltage_deviations', np.zeros(self.NUM_EVCS)), dtype=np.float32)
            self.cumulative_deviation = float(info_dict.get('cumulative_deviation', 0.0))
            self.attack_active = bool(info_dict.get('attack_active', False))
            self.target_evcs = np.asarray(info_dict.get('target_evcs', np.zeros(self.NUM_EVCS)))
            self.attack_duration = float(info_dict.get('attack_duration', 0.0))
            
            # Get reward
            reward = float(rewards[self.agent_type] if isinstance(rewards, dict) else rewards)
            
            return self.state, reward, done, truncated, info
            
        except Exception as e:
            print(f"Error in SACWrapper step: {e}")
            print(f"Action shape: {action.shape if isinstance(action, np.ndarray) else type(action)}")
            if 'dqn_action' in locals():
                print(f"DQN action shape: {dqn_action.shape}, type: {type(dqn_action)}")
            
            # Return the stored tracking variables in case of error
            return self.state, 0.0, True, False, {
                'voltage_deviations': current_tracking['voltage_deviations'],
                'cumulative_deviation': current_tracking['cumulative_deviation'],
                'attack_active': current_tracking['attack_active'],
                'target_evcs': current_tracking['target_evcs'],
                'attack_duration': current_tracking['attack_duration'],
                'error': str(e)
            }
        
    def update_agents(self, dqn_agent= None, sac_defender=None, sac_attacker= None):
        """Update the agents used by the wrapper."""
        if dqn_agent is not None:
            self.dqn_agent = dqn_agent
            print("Updated DQN agent")
        if sac_defender is not None:
            self.sac_defender = sac_defender
            print("Updated SAC defender")
        if sac_attacker is not None:
            self.sac_attacker = sac_attacker
            print("Updated SAC attacker")

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        try:
            # Reset the environment
            obs_info = self.env.reset(seed=seed)
            
            # Handle different return formats
            if isinstance(obs_info, tuple):
                obs, info = obs_info
            else:
                obs = obs_info
                info = {}
            
            # Convert observation to numpy array
            self.state = np.asarray(obs, dtype=np.float32)
            
            # Reset tracking variables
            self.voltage_deviations = np.zeros(self.NUM_EVCS, dtype=np.float32)
            self.cumulative_deviation = 0.0
            self.attack_active = False
            self.target_evcs = np.zeros(self.NUM_EVCS)
            self.attack_duration = 0.0
            
            # Return observation and info dict according to Gym API
            return self.state, info
            
        except Exception as e:
            print(f"Error in SACWrapper reset: {e}")
            
            # Return zero observation and empty info on error
            self.state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return self.state, {
                'error': str(e),
                'voltage_deviations': self.voltage_deviations,
                'cumulative_deviation': self.cumulative_deviation,
                'attack_active': self.attack_active,
                'target_evcs': self.target_evcs,
                'attack_duration': self.attack_duration
            }

    # def decode_dqn_action(self, action):
    #     """Decode DQN action by delegating to the underlying environment."""
    #     if hasattr(self.env, 'decode_dqn_action'):
    #         return self.env.decode_dqn_action(action)
    #     elif hasattr(self.env, 'decode_action'):  # Fallback to decode_action if available
    #         return self.env.decode_action(action)
    #     else:
    #         raise AttributeError("Neither decode_dqn_action nor decode_action method found in environment")

class EVCS_PowerSystem_PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.all_weights = []

        # Initial dense layers
        self.dense1 = tf.keras.layers.Dense(
            256,
            activation='tanh',
            kernel_initializer='glorot_normal',
            name='dense1'
        )
        self.all_weights.extend(self.dense1.trainable_weights)

        # Reshape layer to prepare for LSTM
        self.reshape = tf.keras.layers.Reshape((1, 256))

        # LSTM layers with proper input shape
        self.lstm1 = tf.keras.layers.LSTM(
            units=512,
            return_sequences=True,  # Changed to True for stacked LSTM
            activation='tanh',
            kernel_initializer='glorot_normal',
            name='lstm1'
        )
        self.all_weights.extend(self.lstm1.trainable_weights)

        self.lstm2 = tf.keras.layers.LSTM(
            units=512,
            return_sequences=True,  # Changed to True for stacked LSTM
            activation='tanh',
            kernel_initializer='glorot_normal',
            name='lstm2'
        )
        self.all_weights.extend(self.lstm2.trainable_weights)

        self.lstm3 = tf.keras.layers.LSTM(
            units=512,
            return_sequences=True,  # Changed to True for stacked LSTM
            activation='tanh',
            kernel_initializer='glorot_normal',
            name='lstm3'
        )
        self.all_weights.extend(self.lstm3.trainable_weights)

        self.lstm4 = tf.keras.layers.LSTM(
            units=512,
            return_sequences=False,  # Last LSTM returns single output
            activation='tanh',
            kernel_initializer='glorot_normal',
            name='lstm4'
        )
        self.all_weights.extend(self.lstm4.trainable_weights)

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            NUM_BUSES * 2 + NUM_EVCS * 18,
            kernel_initializer='glorot_normal',
            name='output_layer'
        )
        self.all_weights.extend(self.output_layer.trainable_weights)

    def get_state(self, t):
        """Extract state information from model outputs"""
        outputs = self.call(t)
        
        # Extract components
        v_d = outputs[:, :NUM_BUSES]
        v_q = outputs[:, NUM_BUSES:2*NUM_BUSES]
        evcs_vars = outputs[:, 2*NUM_BUSES:]
        
        # Create state vector
        state = tf.concat([
            v_d,  # Voltage d-axis components
            v_q,  # Voltage q-axis components
            tf.sqrt(v_d**2 + v_q**2),  # Voltage magnitudes
            evcs_vars  # EVCS-specific variables
        ], axis=1)
        
        return state

    def call(self, t):
        # Initial transformation
        x = self.dense1(t)

        # Reshape for LSTM (batch_size, timesteps, features)
        x = self.reshape(x)

        # LSTM processing
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm4(x)

        # Output layer
        output = self.output_layer(x)

        # Split output into different components
        num_voltage_outputs = NUM_BUSES * 2  # For V and theta

        # Separate handling for different outputs
        voltage_magnitude = output[:, :NUM_BUSES]
        voltage_angle = output[:, NUM_BUSES:2*NUM_BUSES]
        evcs_outputs = output[:, 2*NUM_BUSES:]

        # Apply appropriate activations
        voltage_magnitude = tf.exp(voltage_magnitude)  # Ensure positive voltage magnitudes
        voltage_angle = tf.math.atan(voltage_angle)   # Bound angles
        evcs_outputs = tf.nn.tanh(evcs_outputs)      # Bound EVCS outputs

        # Concatenate outputs
        return tf.concat([voltage_magnitude, voltage_angle, evcs_outputs], axis=1)

    @property
    def trainable_variables(self):
        return self.all_weights
@tf.custom_gradient
def safe_op(x):

    # Modified implementation to avoid breaking the computational graph:
    y = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x) + 1e-30) # Add a small value instead of zeros

    def grad(dy):

        # Modified implementation to avoid breaking the computational graph:
        return tf.where(tf.math.is_finite(dy), dy, tf.zeros_like(dy) + 1e-30)  # Add a small value instead of zeros
    return y, grad


def safe_op(tensor_op):
    """Safely perform tensor operations with proper error handling."""
    try:
        result = tf.convert_to_tensor(tensor_op)
        if result is None:
            return tf.constant(0.0, dtype=tf.float32)
        return result
    except Exception as e:
        tf.print(f"Error in safe_op: {e}")
        return tf.constant(0.0, dtype=tf.float32)

@tf.function
def safe_matrix_operations(func):
    """Decorator for safe matrix operations with logging"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Handle tuple return type properly
            if isinstance(result, tuple):
                nan_check = tf.reduce_any([tf.reduce_any(tf.math.is_nan(r)) for r in result])
                if nan_check:
                    tf.print(f"Warning: NaN detected in {func.__name__}")
                    tf.print(f"Input shapes: {[tf.shape(arg) for arg in args]}")
                    return tuple(tf.zeros_like(r) for r in result)
                return result
            else:
                if tf.reduce_any(tf.math.is_nan(result)):
                    tf.print(f"Warning: NaN detected in {func.__name__}")
                    tf.print(f"Input shapes: {[tf.shape(arg) for arg in args]}")
                    return tf.zeros_like(result)
                return result
        except Exception as e:
            tf.print(f"Error in {func.__name__}: {str(e)}")
            batch_size = tf.shape(args[0])[0]
            num_buses = tf.shape(args[0])[1]
            return (tf.zeros([batch_size, num_buses]), tf.zeros([batch_size, num_buses]), {})
    return wrapper

@tf.function
def calculate_power_flow_base(v_d, v_q, G, B, bus_mask):
    """Base power flow calculation with proper shape handling."""
    try:
        # Ensure inputs are rank 2 [batch_size, num_buses]
        v_d = tf.reshape(v_d, [-1, tf.shape(v_d)[-1]])  # [batch, buses]
        v_q = tf.reshape(v_q, [-1, tf.shape(v_q)[-1]])  # [batch, buses]
        
        # Matrix multiplication for power calculations
        # P = V_d * (G * V_d + B * V_q) + V_q * (G * V_q - B * V_d)
        G_vd = tf.matmul(G, tf.expand_dims(v_d, -1))  # [buses, batch, 1]
        G_vq = tf.matmul(G, tf.expand_dims(v_q, -1))  # [buses, batch, 1]
        B_vd = tf.matmul(B, tf.expand_dims(v_d, -1))  # [buses, batch, 1]
        B_vq = tf.matmul(B, tf.expand_dims(v_q, -1))  # [buses, batch, 1]
        
        # Calculate P and Q
        P = v_d * tf.squeeze(G_vd, -1) + v_q * tf.squeeze(G_vq, -1)  # [batch, buses]
        Q = v_d * tf.squeeze(B_vd, -1) - v_q * tf.squeeze(B_vq, -1)  # [batch, buses]
        
        # Apply mask
        P = P * bus_mask  # [batch, buses]
        Q = Q * bus_mask  # [batch, buses]
        
        tf.debugging.assert_shapes([
            (P, ('batch', 'buses')),
            (Q, ('batch', 'buses'))
        ])
        
        return P, Q
        
    except Exception as e:
        tf.print("\nERROR in calculate_power_flow_base:")
        tf.print(str(e))
        tf.print("Error type:", type(e).__name__)
        return None, None, {}
@tf.function
def calculate_power_flow_pcc(v_d, v_q, G, B):
    """PCC power flow calculation."""
    num_buses = tf.shape(v_d)[-1]
    mask = tf.concat([[1.0], tf.zeros(num_buses - 1)], axis=0)
    mask = tf.expand_dims(mask, 0)  # [1, num_buses]
    return calculate_power_flow_base(v_d, v_q, G, B, mask)

@tf.function
def calculate_power_flow_load(v_d, v_q, G, B):
    """Load bus power flow calculation."""
    num_buses = tf.shape(v_d)[-1]
    mask = tf.ones([1, num_buses])
    mask = tf.tensor_scatter_nd_update(mask, [[0, 0]], [0.0])  # Zero out PCC bus
    # Zero out EVCS buses
    for bus in EVCS_BUSES:
        mask = tf.tensor_scatter_nd_update(mask, [[0, bus]], [0.0])
    return calculate_power_flow_base(v_d, v_q, G, B, mask)

@tf.function
def calculate_power_flow_ev(v_d, v_q, G, B):
    """EV bus power flow calculation."""
    num_buses = tf.shape(v_d)[-1]
    mask = tf.zeros([1, num_buses])
    # Set EVCS buses to 1
    for bus in EVCS_BUSES:
        mask = tf.tensor_scatter_nd_update(mask, [[0, bus]], [1.0])
    return calculate_power_flow_base(v_d, v_q, G, B, mask)


def physics_loss(model, t, Y_bus_tf, bus_data, attack_actions, defend_actions):
    """Calculate physics-based losses with proper gradient handling."""
    try:
        # Convert inputs to tensors if they aren't already
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        Y_bus_tf = tf.cast(Y_bus_tf, tf.float32)
        attack_actions = tf.convert_to_tensor(attack_actions, dtype=tf.float32)
        defend_actions = tf.convert_to_tensor(defend_actions, dtype=tf.float32)
        
        # Extract attack and defense actions first
        fdi_voltage = tf.reshape(attack_actions[:, :NUM_EVCS], [-1, NUM_EVCS])
        fdi_current_d = tf.reshape(attack_actions[:, NUM_EVCS:], [-1, NUM_EVCS])
        KP_VOUT = tf.reshape(defend_actions[:, :NUM_EVCS], [-1, NUM_EVCS])
        KI_VOUT = tf.reshape(defend_actions[:, NUM_EVCS:], [-1, NUM_EVCS])

        with tf.GradientTape(persistent=True) as main_tape:
            main_tape.watch(t)
            
            # Get predictions from model
            predictions = model(t)
            
            # Extract predictions
            V = safe_op(tf.exp(predictions[:, :NUM_BUSES]))
            theta = safe_op(tf.math.atan(predictions[:, NUM_BUSES:2*NUM_BUSES]))
            evcs_vars = predictions[:, 2*NUM_BUSES:]
            
            # Calculate voltage components
            v_d = tf.cast(V * tf.cos(theta), tf.float32)
            v_q = tf.cast(V * tf.sin(theta), tf.float32)
            
            # Split Y_bus into real and imaginary parts
            G = tf.cast(tf.math.real(Y_bus_tf), tf.float32)
            B = tf.cast(tf.math.imag(Y_bus_tf), tf.float32)
            
            # Calculate power flows
            P_g_pcc, Q_g_pcc = calculate_power_flow_pcc(v_d, v_q, G, B)
            P_g_load, Q_g_load = calculate_power_flow_load(v_d, v_q, G, B)
            P_g_ev_load, Q_g_ev_load = calculate_power_flow_ev(v_d, v_q, G, B)
            
            # Calculate power mismatches
            P_mismatch = P_g_pcc - (P_g_load + P_g_ev_load)
            Q_mismatch = Q_g_pcc - (Q_g_load + Q_g_ev_load)
            
            # Calculate power flow loss
            power_flow_loss = safe_op(tf.reduce_mean(tf.square(P_mismatch) + tf.square(Q_mismatch)))
            
            # Initialize EVCS losses list and WAC variables
            evcs_loss = []
            wac_error_vdc = tf.zeros_like(t)
            wac_integral_vdc = tf.zeros_like(t)
            wac_error_vout = tf.zeros_like(t)
            wac_integral_vout = tf.zeros_like(t)
            
            # Process each EVCS
            for i, bus in enumerate(EVCS_BUSES):
                try:
                    # Extract EVCS variables
                    evcs = evcs_vars[:, i*18:(i+1)*18]
                    v_ac, i_ac, v_dc, i_dc, v_out, i_out, i_L1, i_L2, v_c, soc, delta, omega, phi_d, phi_q, gamma_d, gamma_q, i_d, i_q = tf.split(evcs, 18, axis=1)
                    
                    # Clarke and Park Transformations
                    v_alpha = v_ac
                    v_beta = tf.zeros_like(v_ac)
                    i_alpha = i_ac
                    i_beta = tf.zeros_like(i_ac)
                    v_out = v_out + fdi_voltage[:, i:i+1]
                    i_d = i_d + fdi_current_d[:, i:i+1]
                    
                    v_d_evcs = safe_op(v_alpha * tf.cos(delta) + v_beta * tf.sin(delta))
                    v_q_evcs = safe_op(-v_alpha * tf.sin(delta) + v_beta * tf.cos(delta))
                    i_d_measured = safe_op(i_alpha * tf.cos(delta) + i_beta * tf.sin(delta))
                    i_q_measured = safe_op(-i_alpha * tf.sin(delta) + i_beta * tf.cos(delta))
                    
                    # Apply FDI attacks
                    v_out += fdi_voltage[:, i:i+1]
                    i_d += fdi_current_d[:, i:i+1]
                    
                    # PLL Dynamics
                    v_q_normalized = tf.nn.tanh(safe_op(v_q_evcs))
                    pll_error = safe_op(EVCS_PLL_KP * v_q_normalized + EVCS_PLL_KI * phi_q)
                    pll_error = tf.clip_by_value(pll_error, -MAX_PLL_ERROR, MAX_PLL_ERROR)
                    
                    # Calculate derivatives with None checks
                    ddelta_dt = main_tape.gradient(delta, t) or tf.zeros_like(delta)
                    domega_dt = main_tape.gradient(omega, t) or tf.zeros_like(omega)
                    dphi_d_dt = main_tape.gradient(phi_d, t) or tf.zeros_like(phi_d)
                    dphi_q_dt = main_tape.gradient(phi_q, t) or tf.zeros_like(phi_q)
                    di_d_dt = main_tape.gradient(i_d, t) or tf.zeros_like(i_d)
                    di_q_dt = main_tape.gradient(i_q, t) or tf.zeros_like(i_q)
                    di_L1_dt = main_tape.gradient(i_L1, t) or tf.zeros_like(i_L1)
                    di_L2_dt = main_tape.gradient(i_L2, t) or tf.zeros_like(i_L2)
                    dv_c_dt = main_tape.gradient(v_c, t) or tf.zeros_like(v_c)
                    dv_dc_dt = main_tape.gradient(v_dc, t) or tf.zeros_like(v_dc)
                    di_out_dt = main_tape.gradient(i_out, t) or tf.zeros_like(i_out)
                    dsoc_dt = main_tape.gradient(soc, t) or tf.zeros_like(soc)

                    P_ac = safe_op(v_d_evcs * i_d + v_q_evcs * i_q)
                    dv_dc_dt_loss = safe_op(tf.reduce_mean(tf.square(dv_dc_dt - (1/(v_dc * C_dc + 1e-6)) * (P_ac - v_dc * i_dc))))

                    modulation_index_vdc = tf.clip_by_value(WAC_KP_VDC * wac_error_vdc + WAC_KI_VDC * wac_integral_vdc, 0, 1)
                    modulation_index_vout = tf.clip_by_value(WAC_KP_VOUT * wac_error_vout + WAC_KI_VOUT * wac_integral_vout, 0, 1)

                    v_out_expected = modulation_index_vout * v_dc
                    v_out_loss = safe_op(tf.reduce_mean(tf.square(v_out - v_out_expected)))

                    v_out_lower = V_OUT_NOMINAL * (1 - V_OUT_VARIATION)
                    v_out_upper = V_OUT_NOMINAL * (1 + V_OUT_VARIATION)
                    v_out_constraint = safe_op(tf.reduce_mean(tf.square(tf.maximum(0.0, v_out_lower - v_out) + tf.maximum(0.0, v_out- v_out_upper))))

                    di_out_dt_loss = safe_op(tf.reduce_mean(tf.square(di_out_dt - (1/L_dc) * (v_out - v_battery - R_battery * i_out))))
                    dsoc_dt_loss = safe_op(tf.reduce_mean(tf.square(dsoc_dt - (EVCS_EFFICIENCY * i_out) / (EVCS_CAPACITY + 1e-6))))

                    P_dc = safe_op(v_dc * i_dc)
                    P_out = safe_op(v_out * i_out)
                    DC_DC_EFFICIENCY = 0.98
                    power_balance_loss = safe_op(tf.reduce_mean(tf.square(P_dc - P_ac) + tf.square(P_out - P_dc * DC_DC_EFFICIENCY)))

                    current_consistency_loss = safe_op(tf.reduce_mean(tf.square(i_ac - i_L2) + tf.square(i_d - i_d_measured) + tf.square(i_q - i_q_measured)))

                    
                    # Calculate EVCS losses with safe handling
                    evcs_losses = [
                        safe_op(tf.reduce_mean(tf.square(ddelta_dt - omega))),
                        safe_op(tf.reduce_mean(tf.square(domega_dt - pll_error))),
                        safe_op(tf.reduce_mean(tf.square(dphi_d_dt - v_d_evcs))),
                        safe_op(tf.reduce_mean(tf.square(dphi_q_dt - v_q_evcs))),
                        safe_op(tf.reduce_mean(tf.square(di_d_dt - (1/LCL_L1) * (v_d_evcs - R * i_d)))),
                        safe_op(tf.reduce_mean(tf.square(di_q_dt - (1/LCL_L1) * (v_q_evcs - R * i_q)))),
                        safe_op(tf.reduce_mean(tf.square(di_L1_dt - (1/LCL_L1) * (v_d_evcs - v_c - R * i_L1)))),
                        safe_op(tf.reduce_mean(tf.square(di_L2_dt - (1/LCL_L2) * (v_c - v_ac - R * i_L2)))),
                        safe_op(tf.reduce_mean(tf.square(dv_c_dt - (1/LCL_CF) * (i_L1 - i_L2)))),
                        safe_op(tf.reduce_mean(tf.square(dv_dc_dt - (1/(v_dc * C_dc + 1e-6)) * (P_ac - v_dc * i_dc)))),
                        safe_op(tf.reduce_mean(tf.square(v_out - v_out_expected))),
                        safe_op(tf.reduce_mean(tf.square(di_out_dt - (1/L_dc) * (v_out - v_battery - R_battery * i_out)))),
                        safe_op(tf.reduce_mean(tf.square(dsoc_dt - (EVCS_EFFICIENCY * i_out) / (EVCS_CAPACITY + 1e-6)))),
                        safe_op(tf.reduce_mean(tf.square(P_dc - P_ac) + tf.square(P_out - P_dc * DC_DC_EFFICIENCY))),
                        safe_op(tf.reduce_mean(tf.square(i_ac - i_L2) + tf.square(i_d - i_d_measured) + tf.square(i_q - i_q_measured)))
                        ]
                    
                    evcs_loss.extend(evcs_losses)

                except Exception as e:
                    tf.print(f"Error in EVCS {i} calculations:", e)
                    # Add zero losses if calculation fails
                    evcs_loss.extend([tf.constant(0.0)] * 15)

            # Calculate final losses
            V_regulation_loss = safe_op(tf.reduce_mean(tf.square(V - tf.ones_like(V))))
            evcs_total_loss = safe_op(tf.reduce_sum(evcs_loss))
            wac_loss = safe_op(tf.reduce_mean(tf.square(wac_error_vdc) + tf.square(wac_error_vout)))
            
            # Combine all losses with safe handling
            total_loss = safe_op(power_flow_loss + evcs_total_loss + wac_loss + V_regulation_loss)
            
            return (
                tf.identity(total_loss),
                tf.identity(power_flow_loss),
                tf.identity(evcs_total_loss),
                tf.identity(wac_loss),
                tf.identity(V_regulation_loss)
            )
            
    except Exception as e:
        tf.print("\nERROR in physics_loss:")
        tf.print(str(e))
        tf.print("Error type:", type(e).__name__)
        return (
            tf.constant(1e6, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32)
        )
            




@tf.function
def train_step(model, optimizer, bus_data_batch, Y_bus_tf, bus_data_tf, attack_actions, defend_actions):
    """Performs a single training step with proper tensor handling"""
    with tf.GradientTape(persistent=True) as tape:
        # Calculate all losses
        total_loss, power_flow_loss, evcs_loss, wac_loss, v_reg_loss = physics_loss(
            model, Y_bus_tf, bus_data_tf,
            attack_actions, defend_actions
        )
        
        # Skip gradient update if we got error values
        if tf.abs(total_loss) >= 1e6:  # Check for error condition
            tf.print("Skipping gradient update due to error in physics_loss")
            return tf.constant(1e6, dtype=tf.float32)
            
        # Calculate gradients using total_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Clean up the tape
    del tape
    
    return tf.keras.backend.get_value(total_loss), {
        'power_flow_loss': tf.keras.backend.get_value(power_flow_loss),
        'evcs_loss': tf.keras.backend.get_value(evcs_loss),
        'wac_loss': tf.keras.backend.get_value(wac_loss),
        'v_reg_loss': tf.keras.backend.get_value(v_reg_loss)
    }

def train_model(initial_model, dqn_agent, sac_attacker, sac_defender, Y_bus_tf, bus_data, epochs=1500, batch_size=256):
    """Train the PINN model with proper data handling."""
    try:
        model = initial_model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # Create environment with necessary data
        env = CompetingHybridEnv(
            pinn_model=model,
            y_bus_tf=Y_bus_tf,
            bus_data=bus_data,
            v_base_lv=V_BASE_DC,
            dqn_agent=dqn_agent,
            num_evcs=NUM_EVCS,
            num_buses=NUM_BUSES,
            time_step=TIME_STEP
        )
        
        # Get bus data from environment
        bus_data_tf = tf.cast(bus_data, tf.float32)
        Y_bus_tf = tf.cast(Y_bus_tf, tf.float32)    
        
        history = {
            'total_loss': [],
            'power_flow_loss': [],
            'evcs_loss': [],
            'wac_loss': [],
            'v_reg_loss': []
        }
        
        for epoch in range(epochs):
            try:
                # Reset environment and get initial state
                reset_result = env.reset()
                
                # Handle different return formats from reset()
                if isinstance(reset_result, tuple):
                    state, _ = reset_result
                else:
                    state = reset_result
                    
                if state is None:
                    print(f"Error: Invalid state in epoch {epoch}")
                    continue
                
                # Get actions from agents
                state_reshaped = state.reshape(1, -1)
                dqn_action = dqn_agent.predict(state_reshaped)[0]
                attack_action = sac_attacker.predict(state_reshaped)[0]
                defend_action = sac_defender.predict(state_reshaped)[0]
                
                # Calculate losses
                try:
                    with tf.GradientTape() as tape:
                        losses = physics_loss(
                            model=model,
                            t=tf.constant([[epoch * TIME_STEP]], dtype=tf.float32),
                            Y_bus_tf=Y_bus_tf,
                            bus_data=bus_data_tf,  # Now properly defined
                            attack_actions=tf.constant(attack_action.reshape(1, -1), dtype=tf.float32),
                            defend_actions=tf.constant(defend_action.reshape(1, -1), dtype=tf.float32)
                        )
                        
                        if not losses or len(losses) != 5:
                            print(f"Invalid losses returned in epoch {epoch}")
                            continue
                            
                        total_loss, pf_loss, ev_loss, wac_loss, v_loss = losses
                        
                        # Update history
                        history['total_loss'].append(float(total_loss))
                        history['power_flow_loss'].append(float(pf_loss))
                        history['evcs_loss'].append(float(ev_loss))
                        history['wac_loss'].append(float(wac_loss))
                        history['v_reg_loss'].append(float(v_loss))
                        
                        # Calculate and apply gradients
                        grads = tape.gradient(total_loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                        
                except Exception as e:
                    print(f"Error in loss calculation for epoch {epoch}: {str(e)}")
                    continue
                
                # Take environment step
                try:
                    next_state, rewards, done, truncated, info = env.step({
                        'dqn': dqn_action,
                        'attacker': attack_action,
                        'defender': defend_action
                    })
                    
                except Exception as e:
                    print(f"Error in environment step for epoch {epoch}: {str(e)}")
                    continue
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue
        
        return model, history
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        return initial_model, None


def evaluate_model_with_three_agents(env, dqn_agent, sac_attacker, sac_defender, num_steps=1500):
    """Evaluate the environment with DQN, SAC attacker, and SAC defender agents."""
    try:
        state, _ = env.reset()
        done = False

        # Initialize tracking variables
        tracking_data = {
            'time_steps': [],
            'cumulative_deviations': [],
            'voltage_deviations': [],
            'attack_active_states': [],
            'target_evcs_history': [],
            'attack_durations': [],
            'dqn_actions': [],
            'sac_attacker_actions': [],
            'sac_defender_actions': [],
            'observations': [],
            'evcs_attack_durations': {i: [] for i in range(env.NUM_EVCS)},
            'attack_counts': {i: 0 for i in range(env.NUM_EVCS)},
            'total_durations': {i: 0 for i in range(env.NUM_EVCS)},
            'rewards': []

        }

        for step in range(num_steps):

            current_time = step * env.TIME_STEP
            
            try:
                # Get and process actions
                dqn_raw = dqn_agent.predict(state, deterministic=True)
                dqn_action_scalar = dqn_raw[0] if isinstance(dqn_raw, tuple) else dqn_raw
                
                if isinstance(dqn_action_scalar, np.ndarray):
                    if dqn_action_scalar.ndim == 0:
                        dqn_action_scalar = int(dqn_action_scalar.item())
                    elif dqn_action_scalar.size == 1:
                        dqn_action_scalar = int(dqn_action_scalar[0])
                
                dqn_action = env.decode_action(dqn_action_scalar)
                sac_attacker_action, _ = sac_attacker.predict(state, deterministic=True)
                sac_defender_action, _ = sac_defender.predict(state, deterministic=True)
                
                action = {
                    'dqn': dqn_action,
                    'attacker': sac_attacker_action,
                    'defender': sac_defender_action
                }

                # Take step
                next_state, rewards, done, truncated, info = env.step(action)


                
                # # Debug prints
                # print(f"\nStep {step} (Time: {current_time:.3f}s):")
                # print(f"DQN Action Scalar: {dqn_action_scalar}, Type: {type(dqn_action_scalar)}")
                # print(f"Decoded DQN Action: {dqn_action}")
                # print(f"Target EVCSs: {info.get('target_evcs', [0] * env.NUM_EVCS)}")
                # print(f"Attack Duration: {info.get('attack_duration', 0)}")
                
                # Store data
                tracking_data['time_steps'].append(current_time)
                tracking_data['cumulative_deviations'].append(info.get('cumulative_deviation', 0))
                tracking_data['voltage_deviations'].append(info.get('voltage_deviations', [0] * env.NUM_EVCS))
                tracking_data['attack_active_states'].append(info.get('attack_active', False))
                tracking_data['target_evcs_history'].append(info.get('target_evcs', [0] * env.NUM_EVCS))
                tracking_data['attack_durations'].append(info.get('attack_duration', 0))
                tracking_data['dqn_actions'].append(dqn_action)
                tracking_data['sac_attacker_actions'].append(sac_attacker_action)
                tracking_data['sac_defender_actions'].append(sac_defender_action)
                tracking_data['observations'].append(next_state)
                tracking_data['rewards'].append(rewards)

                # Track EVCS-specific attack data
                target_evcs = info.get('target_evcs', [0] * env.NUM_EVCS)
                attack_duration = info.get('attack_duration', 0)
                for i in range(env.NUM_EVCS):
                    if target_evcs[i] == 1:
                        tracking_data['evcs_attack_durations'][i].append(attack_duration)
                        tracking_data['attack_counts'][i] += 1
                        tracking_data['total_durations'][i] += attack_duration
                
                state = next_state

                if done:
                    break

            except Exception as e:
                print(f"Error in evaluation step {step}: {str(e)}")
                continue

        # Calculate average attack durations
        avg_attack_durations = []
        for i in range(env.NUM_EVCS):
            if tracking_data['attack_counts'][i] > 0:
                avg_duration = tracking_data['total_durations'][i] / tracking_data['attack_counts'][i]
            else:
                avg_duration = 0
            avg_attack_durations.append(avg_duration)

        # Convert lists to numpy arrays and add calculated metrics
        processed_data = {}
        for key in tracking_data:
            if isinstance(tracking_data[key], dict):
                processed_data[key] = tracking_data[key]
            elif len(tracking_data[key]) > 0:
                processed_data[key] = np.array(tracking_data[key])
        
        processed_data['avg_attack_durations'] = np.array(avg_attack_durations)

        return processed_data

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None, None

def check_constraints(state, info):
        """Helper function to check individual constraints."""
        violations = []
        
        # Extract relevant state components
        # Assuming state structure matches your environment's observation space
        voltage_indices = slice(0, NUM_BUSES)  # Adjust based on your state structure
        current_indices = slice(NUM_BUSES, 2*NUM_BUSES)  # Adjust as needed
        
        # Check voltage constraints (0.9 to 1.1 p.u.)
        voltages = state[voltage_indices]
        if np.any(voltages < 0.8) or np.any(voltages > 1.2):
            violations.append({
                'type': 'Voltage',
                'values': voltages,
                'limits': (0.8, 1.2),
                'violated_indices': np.where((voltages < 0.8) | (voltages > 1.2))[0]
            })

        # Check current constraints (-1.0 to 1.0 p.u.)
        currents = state[current_indices]
        if np.any(np.abs(currents) > 1.0):
            violations.append({
                'type': 'Current',
                'values': currents,
                'limits': (-1.0, 1.0),
                'violated_indices': np.where(np.abs(currents) > 1.0)[0]
            })

        # Check power constraints if available in state
        if 'power_output' in info:
            power = info['power_output']
            if np.any(np.abs(power) > 1.0):
                violations.append({
                    'type': 'Power',
                    'values': power,
                    'limits': (-1.0, 1.0),
                    'violated_indices': np.where(np.abs(power) > 1.0)[0]
                })

        # Check SOC constraints if available
        if 'soc' in info:
            soc = info['soc']
            if np.any((soc < 0.1) | (soc > 0.9)):
                violations.append({
                    'type': 'State of Charge',
                    'values': soc,
                    'limits': (0.1, 0.9),
                    'violated_indices': np.where((soc < 0.1) | (soc > 0.9))[0]
                })

        return violations, info

def validate_physics_constraints(env, dqn_agent, sac_attacker, sac_defender, num_episodes=5):
    """Validate that the agents respect physics constraints with detailed reporting."""


    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            try:
                # Get actions from all agents
                dqn_action_scalar = dqn_agent.predict(state, deterministic=True)[0]
                dqn_action = env.decode_dqn_action(dqn_action_scalar)
                attacker_action = sac_attacker.predict(state, deterministic=True)[0]
                defender_action = sac_defender.predict(state, deterministic=True)[0]
                
                # Combine actions
                action = {
                    'dqn': dqn_action,
                    'attacker': attacker_action,
                    'defender': defender_action
                }
                
                # Take step in environment
                next_state, rewards, done, truncated, info = env.step(action)
                
                # Check for physics violations
                violations = check_constraints(next_state, info)
                
                if violations:
                    print(f"\nPhysics constraints violated in episode {episode}, step {step_count}:")
                    for violation in violations:
                        print(f"\nViolation Type: {violation['type']}")
                        print(f"Limits: {violation['limits']}")
                        # print(f"Violated at indices: {violation['violated_indices']}")
                        # print(f"Values at violated indices: {violation['values'][violation['violated_indices']]}")
                    return False
                
                state = next_state
                step_count += 1
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                return False
            
    print("All physics constraints validated successfully!")
    return True, info





def plot_evaluation_results(results, save_dir="./figures"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract data from results
    time_steps = results['time_steps']
    cumulative_deviations = results['cumulative_deviations']
    voltage_deviations = np.array(results['voltage_deviations'])
    attack_active_states = results['attack_active_states']
    avg_attack_durations = results['avg_attack_durations']

    # Plot cumulative deviations over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, cumulative_deviations, label='Cumulative Deviations')
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Deviations')
    plt.title('Cumulative Deviations Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/cumulative_deviations_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, results['rewards'], label='rewards') # Fixed to use results dictionary
    plt.xlabel('Time (s)')
    plt.ylabel('Rewards from Joint Environment')
    plt.title('Rewards Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/rewards_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()



    # Plot voltage deviations for each EVCS over time
    plt.figure(figsize=(12, 6))
    for i in range(voltage_deviations.shape[1]):
        plt.plot(time_steps, voltage_deviations[:, i], label=f'EVCS {i+1} Voltage Deviation')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage Deviation (p.u.)')
    plt.title('Voltage Deviations Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/voltage_deviations_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot attack active states over time
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, attack_active_states, label='Attack Active State')
    plt.xlabel('Time (s)')
    plt.ylabel('Attack Active State')
    plt.title('Attack Active State Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/attack_states_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot average attack durations for each EVCS
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(avg_attack_durations)), avg_attack_durations, tick_label=[f'EVCS {i+1}' for i in range(len(avg_attack_durations))])
    plt.xlabel('EVCS')
    plt.ylabel('Average Attack Duration (s)')
    plt.title('Average Attack Duration for Each EVCS')
    plt.grid(True)
    plt.savefig(f"{save_dir}/avg_attack_durations_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()




if __name__ == '__main__':
    # Define physics parameters
    physics_params = {
        'voltage_limits': (0.8, 1.2),
        'v_out_nominal': 1.0,
        'current_limits': (-1.0, 1.0),
        'i_rated': 1.0,
        'attack_magnitude': 0.01,
        'current_magnitude': 0.03,
        'wac_kp_limits': (0.0, 2.0),
        'wac_ki_limits': (0.0, 2.0),
        'control_saturation': 0.3,
        'power_limits': (-1.0, 1.0),
        'power_ramp_rate': 0.1,
        'evcs_efficiency': 0.98,
        'soc_limits': (0.1, 0.9)
    }

    # Initialize the PINN model
    initial_pinn_model = EVCS_PowerSystem_PINN()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{timestamp}"
    model_dir = f"./models/{timestamp}"
    for dir_path in [log_dir, model_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Create the Discrete Environment for DQN Agent
    print("Creating the DiscreteHybridEnv environment...")
    discrete_env = DiscreteHybridEnv(
        pinn_model=initial_pinn_model,
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP,
        **physics_params
    )

    # Initialize callbacks
    
    dqn_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/dqn_checkpoints/",
        name_prefix="dqn"
    )
    
    # Initialize the DQN Agent with improved parameters
    print("Initializing the DQN agent...")
    dqn_agent = DQN(
        'MlpPolicy',
        discrete_env,
        verbose=1,
        learning_rate=3e-3,
        buffer_size=10000,
        exploration_fraction=0.3,
        exploration_final_eps=0.2,
        train_freq=4,
        batch_size=32,
        gamma=0.99,
        device='cuda',
        tensorboard_log=f"{log_dir}/dqn/"
    )

    # Train DQN with monitoring
    print("Training DQN agent...")
    dqn_agent.learn(
        total_timesteps=1000,
        callback=dqn_checkpoint,
        progress_bar=True
    )
    dqn_agent.save(f"{model_dir}/dqn_final")

    # Create the CompetingHybridEnv
    print("Creating the CompetingHybridEnv environment...")
    combined_env = CompetingHybridEnv(
        pinn_model=initial_pinn_model,
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        dqn_agent=dqn_agent,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP,
        **physics_params
    )

    print("Creating SAC Wrapper environments...")
    sac_attacker_env = SACWrapper(
        env=combined_env,
        agent_type='attacker',
        dqn_agent=dqn_agent
    )
    # Initialize SAC Attacker
    print("Initializing SAC Attacker...")
    sac_attacker = SAC(
        'MlpPolicy',
        sac_attacker_env,
        verbose=1,
        learning_rate=5e-4,
        buffer_size=10000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        device='cuda',
        tensorboard_log=f"{log_dir}/sac_attacker/"
    )

    # Create defender wrapper environment with the trained attacker
    print("Creating SAC Defender environment...")
    sac_defender_env = SACWrapper(
        env=combined_env,
        agent_type='defender',
        dqn_agent=dqn_agent
    )

    # Initialize SAC Defender
    print("Initializing SAC Defender...")
    sac_defender = SAC(
        'MlpPolicy',
        sac_defender_env,
        verbose=1,
        learning_rate=1e-7,
        buffer_size=1000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
        device='cuda',
        tensorboard_log=f"{log_dir}/sac_defender/"
    )

    # Update wrapper environments with both agents
    sac_attacker_env.sac_defender = sac_defender
    sac_defender_env.sac_attacker = sac_attacker

    # Create callbacks for monitoring
    sac_attacker_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/sac_attacker_checkpoints/",
        name_prefix="attacker"
    )
    
    sac_defender_checkpoint = CheckpointCallback(
        save_freq=1000,
        save_path=f"{model_dir}/sac_defender_checkpoints/",
        name_prefix="defender"
    )
# New Addition 
    print("Training the SAC Attacker agent...")
    sac_attacker.learn(
        total_timesteps=500,
        callback=sac_attacker_checkpoint,
        progress_bar=True
    )
    sac_attacker.save(f"{model_dir}/sac_attacker_final")

    print("Training the SAC Defender agent...")
    sac_defender.learn(
        total_timesteps=500,
        callback=sac_defender_checkpoint,
        progress_bar=True
    )
    sac_defender.save(f"{model_dir}/sac_defender_final")

    num_iterations = 2
    # Joint training loop with validation
    print("Starting joint training...")
    for iteration in range(num_iterations):
        print(f"\nJoint training iteration {iteration + 1}/{num_iterations}")
        
        # Train agents with progress monitoring
        for agent, name, callback, env in [
            (dqn_agent, "DQN", dqn_checkpoint, discrete_env),
            (sac_attacker, "SAC Attacker", sac_attacker_checkpoint, sac_attacker_env),
            (sac_defender, "SAC Defender", sac_defender_checkpoint, sac_defender_env)
        ]:
            print(f"\nTraining {name}...")
            if name == "SAC Defender":
                total_timesteps=500
            else:
                total_timesteps=500
            agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True
            )
            agent.save(f"{model_dir}/{name.lower()}_iter_{iteration + 1}")

            # Update environment references after each agent training
            combined_env.update_agents(dqn_agent, sac_attacker, sac_defender)
            sac_attacker_env.update_agents(sac_defender=sac_defender, dqn_agent=dqn_agent)
            sac_defender_env.update_agents(sac_attacker=sac_attacker, dqn_agent=dqn_agent)

        # # Validate physics constraints
        # print("\nValidating physics constraints...")
        # validation_success = validate_physics_constraints(
        #     combined_env,
        #     dqn_agent,
        #     sac_attacker,
        #     sac_defender,
        #     num_episodes=3
        # )
        # print(f"Physics validation: {'Passed' if validation_success else 'Failed'}")


    print("Training the PINN model with the hybrid RL agents (DQN for target, SAC Attacker for FDI, and SAC Defender for stabilization)...")
    trained_pinn_model, training_history = train_model(
        initial_model=initial_pinn_model,
        dqn_agent=dqn_agent,
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        Y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        epochs=20,
        batch_size=128
        )

    # Optionally plot training history
    if training_history is not None:
        plt.figure(figsize=(12, 8))
        plt.plot(training_history['total_loss'], label='Total Loss')
        plt.plot(training_history['power_flow_loss'], label='Power Flow Loss')
        plt.plot(training_history['evcs_loss'], label='EVCS Loss')
        plt.plot(training_history['wac_loss'], label='WAC Loss')
        plt.plot(training_history['v_reg_loss'], label='V Regulation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()

        # After training the PINN model, create a new environment using the trained model
    print("Creating a new CompetingHybridEnv environment with the trained PINN model...")
    trained_combined_env = CompetingHybridEnv(
        pinn_model=trained_pinn_model,  # Use the trained PINN model here
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        dqn_agent=dqn_agent,  # Use the trained agents
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP
    )

        # Update the environment's agent references if necessary
    trained_combined_env.sac_attacker = sac_attacker
    trained_combined_env.sac_defender = sac_defender
    trained_combined_env.dqn_agent = dqn_agent

    # Final evaluation
    print("\nRunning final evaluation...")
    results = evaluate_model_with_three_agents(
        env=trained_combined_env,
        dqn_agent=dqn_agent,
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        num_steps=1000
    )

    # Convert NumPy arrays to lists before saving
    serializable_results = {
        'time_steps': results['time_steps'].tolist(),
        'cumulative_deviations': results['cumulative_deviations'].tolist(),
        'voltage_deviations': [vd.tolist() for vd in results['voltage_deviations']],
        'attack_active_states': results['attack_active_states'].tolist(),
        'target_evcs_history': [targets.tolist() if isinstance(targets, np.ndarray) else targets 
                                for targets in results['target_evcs_history']],
        'attack_durations': results['attack_durations'].tolist(),
        'observations': [obs.tolist() for obs in results['observations']],
        'avg_attack_durations': results['avg_attack_durations'].tolist(),
        'rewards': results['rewards'].tolist()
    }

    # Assuming 'results' is the dictionary returned from evaluate_model_with_three_agents
    plot_evaluation_results(serializable_results)

    print("\nTraining completed successfully!")
