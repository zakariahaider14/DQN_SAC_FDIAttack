# !#Testing with WAC
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import CosineDecay
import csv
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import json


import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC, DQN

from DiscreteHybridEnv import DiscreteHybridEnv
from combined_pinn import CompetingHybridEnv



import os
import tempfile

# Set a different temporary directory
os.environ['TMPDIR'] = tempfile.gettempdir()
os.environ['TORCH_HOME'] = tempfile.gettempdir()

# Disable PyTorch's JIT compilation
os.environ['PYTORCH_JIT'] = '0'

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
torch.set_num_threads(1)



# import shimmy
# Check if TensorFlow can see the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



# System Constants
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

EVCS_OUTER_KP = 0.5 #0.5 and 0.3 was original value
EVCS_OUTER_KI = 0.3

EVCS_INNER_KP = 0.5
EVCS_INNER_KI = 0.3
OMEGA_N = 2 * np.pi * 60  # Nominal angular frequency (60 Hz)

# Wide Area Controller Parameters
WAC_KP_VDC = 0.1
WAC_KI_VDC = 0.05

WAC_KP_VOUT = 0.1
WAC_KI_VOUT = 0.05 # original value is 0.5
WAC_VOLTAGE_SETPOINT = V_BASE_DC / V_BASE_LV  # Desired DC voltage in p.u.
WAC_VOUT_SETPOINT = V_BASE_DC / V_BASE_LV  # Desired output voltage in p.u.


# Other Parameters
CONSTRAINT_WEIGHT = 1.0
LCL_L1 = 0.001 / Z_BASE_LV  # Convert to p.u.
LCL_L2 = 0.001 / Z_BASE_LV  # Convert to p.u.
LCL_CF = 10e-1 * S_BASE / (V_BASE_LV**2)  # Convert to p.u.
R = 0.01 / Z_BASE_LV  # Convert to p.u.
C_dc = 0.01 * S_BASE / (V_BASE_LV**2)  # Convert to p.u.    


L_dc = 0.001 / Z_BASE_LV  # Convert to p.u.
v_battery = 800 / V_BASE_DC  # Convert to p.u.
R_battery = 0.01 / Z_BASE_LV  # Convert to p.u.

# Time parameters
TIME_STEP = 0.1 # 1 ms
TOTAL_TIME = 10000  # 100 seconds

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

bus_data[:, 1:3] = bus_data[:, 1:3]*1e3 / S_BASE
EVCS_CAPACITY = 80e3 / S_BASE    # Convert kW to per-unit


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

class SACWrapper(gym.Env):
    """Wrapper for SAC agents to interact with the environment."""
    
    def __init__(self, env, agent_type, dqn_agent=None, sac_defender=None, sac_attacker=None):
        """Initialize the SAC wrapper.
        
        Args:
            env: The environment to wrap
            agent_type: Either 'attacker' or 'defender'
            dqn_agent: The DQN agent
            sac_defender: The SAC defender agent (optional)
            sac_attacker: The SAC attacker agent (optional)
        """
        super(SACWrapper, self).__init__()
        
        self.env = env
        self.agent_type = agent_type
        self.dqn_agent = dqn_agent
        self.sac_defender = sac_defender
        self.sac_attacker = sac_attacker
        self.state = None
        
        # Set action and observation spaces based on agent type
        if agent_type == 'attacker':
            self.action_space = env.sac_attacker_action_space
        elif agent_type == 'defender':
            self.action_space = env.sac_defender_action_space
        else:
            raise ValueError(f"Invalid agent_type: {agent_type}. Must be either 'attacker' or 'defender'")
        
        self.observation_space = env.observation_space

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        try:
            result = self.env.reset(seed=seed)
            if isinstance(result, tuple):
                observation, info = result
            else:
                print("Warning: Environment reset didn't return info dict")
                observation = result
                info = {}
            
            self.state = observation
            return observation, info
            
        except Exception as e:
            print(f"Error in SACWrapper reset: {e}")
            default_state = np.zeros(self.observation_space.shape)
            default_info = {'error': str(e)}
            return default_state, default_info

    def step(self, action):
        """Take a step in the environment."""
        try:
            if self.agent_type == 'attacker':
                # Get DQN action and decode it
                dqn_state = np.array(self.state).reshape(1, -1)  # Ensure proper shape
                dqn_action_scalar = self.dqn_agent.predict(dqn_state, deterministic=True)[0]
                dqn_action = self.env.decode_dqn_action(dqn_action_scalar)
                
                # Get defender action if available
                defender_action = (self.sac_defender.predict(self.state, deterministic=True)[0] 
                                 if self.sac_defender else np.zeros(self.env.NUM_EVCS * 2))
                
                # Ensure actions are properly shaped
                action = np.array(action).reshape(-1)
                defender_action = np.array(defender_action).reshape(-1)
                
                combined_action = {
                    'dqn': dqn_action,
                    'attacker': action,
                    'defender': defender_action
                }
            else:  # defender
                # Get DQN action and decode it
                dqn_state = np.array(self.state).reshape(1, -1)
                dqn_action_scalar = self.dqn_agent.predict(dqn_state, deterministic=True)[0]
                dqn_action = self.env.decode_dqn_action(dqn_action_scalar)
                
                # Get attacker action if available
                attacker_action = (self.sac_attacker.predict(self.state, deterministic=True)[0] 
                                 if self.sac_attacker else np.zeros(self.env.NUM_EVCS * 2))
                
                # Ensure actions are properly shaped
                action = np.array(action).reshape(-1)
                attacker_action = np.array(attacker_action).reshape(-1)
                
                combined_action = {
                    'dqn': dqn_action,
                    'attacker': attacker_action,
                    'defender': action
                }
            
            # Take step in environment
            next_state, rewards, done, truncated, info = self.env.step(combined_action)
            
            # Ensure state is properly shaped numpy array
            self.state = np.array(next_state)
            
            # Return appropriate reward based on agent type
            return (
                self.state,
                rewards[self.agent_type] if isinstance(rewards, dict) else rewards,
                done,
                truncated,
                info
            )
            
        except Exception as e:
            print(f"Error in SACWrapper step: {e}")
            return (
                self.state,
                0.0,
                True,
                False,
                {'error': str(e)}
            )

    def update_agents(self, sac_defender=None, sac_attacker=None, dqn_agent=None):
        """Update agent references."""
        if sac_defender is not None:
            self.sac_defender = sac_defender
        if sac_attacker is not None:
            self.sac_attacker = sac_attacker
        if dqn_agent is not None:
            self.dqn_agent = dqn_agent
        return True

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()


class EVCS_PowerSystem_PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.all_weights = []

        # Initial layer
        self.initial_dense = tf.keras.layers.Dense(
            units=500,
            activation='tanh',
            kernel_initializer='glorot_normal',
            name="initial_dense"
        )

        # Dense layers
        self.dense_layers = []
        for i in range(6):
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    units=500,
                    activation='tanh',
                    kernel_initializer='glorot_normal',
                    name=f"dense_layer_{i}"
                )
            )

        # Output layer
        self.output_dense = tf.keras.layers.Dense(
            units=NUM_BUSES * 2 + NUM_EVCS * 18,
            activation=None,  # Linear activation for output
            kernel_initializer='glorot_normal',
            name="output_dense"
        )

    def call(self, t):
        x = self.initial_dense(t)
        
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            
        return self.output_dense(x)

    @property
    def trainable_variables(self):
        variables = (
            self.initial_dense.trainable_variables +
            sum([layer.trainable_variables for layer in self.dense_layers], []) +
            self.output_dense.trainable_variables
        )
        return variables


@tf.custom_gradient
def safe_op(x):
    # Original implementation:
    # y = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

    # Modified implementation to avoid breaking the computational graph:
    y = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x) + 1e-30) # Add a small value instead of zeros

    def grad(dy):
        # Original implementation:
        # return tf.where(tf.math.is_finite(dy), dy, tf.zeros_like(dy))

        # Modified implementation to avoid breaking the computational graph:
        return tf.where(tf.math.is_finite(dy), dy, tf.zeros_like(dy) + 1e-30)  # Add a small value instead of zeros
    return y, grad

def physics_loss(model, t, Y_bus_tf, bus_data, attack_actions, defend_actions):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        predictions = model(t)

        # Split the attack actions for voltage and current manipulation
        fdi_voltage = attack_actions[:, :5]  # First 5 elements are for voltage FDI
        fdi_current_d = attack_actions[:, 5:10]  # Next 5 elements are for d-axis current FDI

        # Split the defender actions for WAC control parameters
        KP_VOUT = defend_actions[:, :5]  # First 5 elements for WAC_KP_VOUT
        KI_VOUT = defend_actions[:, 5:10]  # Next 5 elements for WAC_KI_VOUT


                # Initialize WAC_KP_VOUT and WAC_KI_VOUT
        WAC_KP_VOUT = tf.zeros_like(KP_VOUT)
        WAC_KI_VOUT = tf.zeros_like(KI_VOUT)

        # === NaN Check for Debugging ===
        if tf.reduce_any(tf.math.is_nan(fdi_voltage)) or tf.reduce_any(tf.math.is_nan(fdi_current_d)):
            tf.print("Warning: NaN values detected in FDI variables - fdi_voltage or fdi_current_d.")
            tf.print(f"fdi_voltage: {fdi_voltage}")
            tf.print(f"fdi_current_d: {fdi_current_d}")

        # Extract variables from predictions
        V = safe_op(tf.exp(predictions[:, :NUM_BUSES]))
        theta = safe_op(tf.math.atan(predictions[:, NUM_BUSES:2*NUM_BUSES]))
        evcs_vars = predictions[:, 2*NUM_BUSES:]

        # Compute power flow equations and mismatches
        V_complex = tf.complex(V * tf.cos(theta), V * tf.sin(theta))
        S = V_complex[:, :, tf.newaxis] * tf.math.conj(tf.matmul(Y_bus_tf, V_complex[:, :, tf.newaxis]))
        P_calc = tf.math.real(S)[:, :, 0]
        Q_calc = tf.math.imag(S)[:, :, 0]
        P_mismatch = P_calc - bus_data[:, 1]
        Q_mismatch = Q_calc - bus_data[:, 2]

        # Voltage regulation for all buses
        V_nominal = 1.0  # Nominal voltage in p.u.
        V_lower = 0.85  # Lower voltage limit
        V_upper = 1.15  # Upper voltage limit
        V_regulation_loss = safe_op(tf.reduce_mean(tf.square(tf.maximum(0.0, V_lower - V) + tf.maximum(0.0, V - V_upper))))

        evcs_loss = []
        wac_error_vdc = tf.zeros_like(t)
        wac_integral_vdc = tf.zeros_like(t)
        wac_error_vout = tf.zeros_like(t)
        wac_integral_vout = tf.zeros_like(t)

        for i, bus in enumerate(EVCS_BUSES):
            evcs = evcs_vars[:, i*18:(i+1)*18]

            v_ac, i_ac, v_dc, i_dc, v_out, i_out, i_L1, i_L2, v_c, soc, \
            delta, omega, phi_d, phi_q, gamma_d, gamma_q, i_d, i_q = tf.split(evcs, num_or_size_splits=18, axis=1)


            # Ensure positive voltages
            v_ac = safe_op(tf.exp(v_ac))
            v_dc = safe_op(tf.exp(v_dc))
            v_out = safe_op(tf.exp(v_out))
            v_c = safe_op(tf.exp(v_c))

            v_out += fdi_voltage[:, i:i+1]  # Add the voltage FDI to the output voltage
            i_d   += fdi_current_d[:, i:i+1]  # Add the d-axis current FDI to the d-axis current

            WAC_KP_VOUT += KP_VOUT[:, i:i+1]  # Add the voltage FDI to the output voltage
            WAC_KI_VOUT += KI_VOUT[:, i:i+1]  # Add the d-axis current FDI to the d-axis current



            # Clarke and Park Transformations
            v_alpha = v_ac
            v_beta = tf.zeros_like(v_ac)
            i_alpha = i_ac
            i_beta = tf.zeros_like(i_ac)

            v_d = safe_op(v_alpha  * tf.cos(delta) + v_beta * tf.sin(delta))
            v_q = safe_op(-v_alpha * tf.sin(delta) + v_beta * tf.cos(delta))
            i_d_measured = safe_op(i_alpha * tf.cos(delta) + i_beta * tf.sin(delta))
            i_q_measured = safe_op(-i_alpha * tf.sin(delta) + i_beta * tf.cos(delta))



            # PLL Dynamics
            v_q_normalized = tf.nn.tanh(safe_op(v_q))
            pll_error = safe_op(EVCS_PLL_KP * v_q_normalized + EVCS_PLL_KI * phi_q)
            pll_error = tf.clip_by_value(pll_error, -MAX_PLL_ERROR, MAX_PLL_ERROR)

            # Wide Area Controller
            v_dc_actual = v_dc * V_BASE_DC/V_BASE_LV
            v_out_actual = v_out * V_BASE_DC/V_BASE_LV




            # Modify the Wide Area Controller part
            wac_error_vdc = WAC_VOLTAGE_SETPOINT - v_dc
            wac_integral_vdc += wac_error_vdc * TIME_STEP
            wac_output_vdc = WAC_KP_VDC * wac_error_vdc + WAC_KI_VDC * wac_integral_vdc
            v_dc_ref = WAC_VOLTAGE_SETPOINT + wac_output_vdc

            wac_error_vout = WAC_VOUT_SETPOINT - v_out
            wac_integral_vout += wac_error_vout * TIME_STEP
            wac_output_vout = WAC_KP_VOUT * wac_error_vout + WAC_KI_VOUT * wac_integral_vout
            v_out_ref = WAC_VOUT_SETPOINT + wac_output_vout

            # Converter Outer Loop
            i_d_ref = safe_op(EVCS_OUTER_KP * (v_dc - v_dc_ref) + EVCS_OUTER_KI * gamma_d)
            i_q_ref = safe_op(EVCS_OUTER_KP * (0 - v_q) + EVCS_OUTER_KI * gamma_q)

            # Converter Inner Loop
            v_d_conv = safe_op(EVCS_INNER_KP * (i_d_ref - i_d) + EVCS_INNER_KI * phi_d - omega * LCL_L1 * i_q + v_d)
            v_q_conv = safe_op(EVCS_INNER_KP * (i_q_ref - i_q) + EVCS_INNER_KI * phi_q + omega * LCL_L1 * i_d + v_q)

            # Calculate losses
            ddelta_dt = tape.gradient(delta, t)
            domega_dt = tape.gradient(omega, t)
            dphi_d_dt = tape.gradient(phi_d, t)
            dphi_q_dt = tape.gradient(phi_q, t)
            di_d_dt = tape.gradient(i_d, t)
            di_q_dt = tape.gradient(i_q, t)
            di_L1_dt = tape.gradient(i_L1, t)
            di_L2_dt = tape.gradient(i_L2, t)
            dv_c_dt = tape.gradient(v_c, t)
            dv_dc_dt = tape.gradient(v_dc, t)
            di_out_dt = tape.gradient(i_out, t)
            dsoc_dt = tape.gradient(soc, t)

            ddelta_dt_loss = safe_op(tf.reduce_mean(tf.square(ddelta_dt - omega)))
            domega_dt_loss = safe_op(tf.reduce_mean(tf.square(domega_dt - pll_error)))
            dphi_d_dt_loss = safe_op(tf.reduce_mean(tf.square(dphi_d_dt - v_d)))
            dphi_q_dt_loss = safe_op(tf.reduce_mean(tf.square(dphi_q_dt - v_q)))

            di_d_dt_loss = safe_op(tf.reduce_mean(tf.square(di_d_dt - (1/LCL_L1) * (v_d_conv - R * i_d - v_d + omega * LCL_L1 * i_q))))
            di_q_dt_loss = safe_op(tf.reduce_mean(tf.square(di_q_dt - (1/LCL_L1) * (v_q_conv - R * i_q - v_q - omega * LCL_L1 * i_d))))

            di_L1_dt_loss = safe_op(tf.reduce_mean(tf.square(di_L1_dt - (1/LCL_L1) * (v_d_conv - v_c - R * i_L1))))
            di_L2_dt_loss = safe_op(tf.reduce_mean(tf.square(di_L2_dt - (1/LCL_L2) * (v_c - v_ac - R * i_L2))))
            dv_c_dt_loss = safe_op(tf.reduce_mean(tf.square(dv_c_dt - (1/LCL_CF) * (i_L1 - i_L2))))

            P_ac = safe_op(v_d * i_d + v_q * i_q)
            dv_dc_dt_loss = safe_op(tf.reduce_mean(tf.square(dv_dc_dt - (1/(v_dc * C_dc + 1e-6)) * (P_ac - v_dc * i_dc))))

            modulation_index_vdc = tf.clip_by_value(WAC_KP_VDC * wac_error_vdc + WAC_KI_VDC * wac_integral_vdc, 0, 1)
            modulation_index_vout = tf.clip_by_value(WAC_KP_VOUT * wac_error_vout + WAC_KI_VOUT * wac_integral_vout, 0, 1)

            
            
            
#             duty_cycle = tf.clip_by_value(v_out_ref / v_dc, 0, 1)
            v_out_expected = modulation_index_vout * v_dc

            v_out_loss = safe_op(tf.reduce_mean(tf.square(v_out - v_out_expected)))

            v_out_actual = v_out * V_BASE_DC/V_BASE_LV
            v_out_lower = V_OUT_NOMINAL * (1 - V_OUT_VARIATION)
            v_out_upper = V_OUT_NOMINAL * (1 + V_OUT_VARIATION)
            v_out_constraint = safe_op(tf.reduce_mean(tf.square(tf.maximum(0.0, v_out_lower - v_out_actual) + tf.maximum(0.0, v_out_actual - v_out_upper))))

            L_dc = 0.001 / Z_BASE_LV  # Convert to p.u.
            v_battery = 800 / V_BASE_DC  # Convert to p.u.
            R_battery = 0.1 / Z_BASE_LV  # Convert to p.u.

            di_out_dt_loss = safe_op(tf.reduce_mean(tf.square(di_out_dt - (1/L_dc) * (v_out - v_battery - R_battery * i_out))))

            dsoc_dt_loss = safe_op(tf.reduce_mean(tf.square(dsoc_dt - (EVCS_EFFICIENCY * i_out) / (EVCS_CAPACITY + 1e-6))))

            P_dc = safe_op(v_dc * i_dc)
            P_out = safe_op(v_out * i_out)
            DC_DC_EFFICIENCY = 0.98
            power_balance_loss = safe_op(tf.reduce_mean(tf.square(P_dc - P_ac) + tf.square(P_out - P_dc * DC_DC_EFFICIENCY)))

            current_consistency_loss = safe_op(tf.reduce_mean(tf.square(i_ac - i_L2) + tf.square(i_d - i_d_measured) + tf.square(i_q - i_q_measured)))

            
            
            
            # Append all losses
            evcs_loss.extend([
                ddelta_dt_loss, domega_dt_loss, dphi_d_dt_loss, dphi_q_dt_loss,
                di_d_dt_loss, di_q_dt_loss, di_L1_dt_loss, di_L2_dt_loss, dv_c_dt_loss,
                dv_dc_dt_loss, v_out_loss, v_out_constraint, di_out_dt_loss, dsoc_dt_loss,
                power_balance_loss, current_consistency_loss
            ])

       # Final loss calculations
        power_flow_loss = safe_op(tf.reduce_mean(tf.square(P_mismatch) + tf.square(Q_mismatch)))
        evcs_total_loss = safe_op(tf.reduce_sum(evcs_loss))
        wac_loss = safe_op(tf.reduce_mean(tf.square(wac_error_vdc) + tf.square(wac_error_vout)))
        V_regulation_loss = safe_op(tf.reduce_mean(V_regulation_loss))

        # Combine into total loss
        total_loss = power_flow_loss + evcs_total_loss + wac_loss + V_regulation_loss

    # Gradient calculations and debugging (no changes needed here)
    variables = model.trainable_variables
    gradients = tape.gradient(total_loss, variables)

    for var, grad in zip(variables, gradients):
        if grad is not None:
            tf.debugging.check_numerics(grad, f"Gradient for variable {var.name} contains NaN or Inf.")
        else:
            tf.print(f"WARNING: Gradient is None for variable {var.name}")

    del tape

    return total_loss, power_flow_loss, evcs_total_loss, wac_loss, V_regulation_loss



@tf.function
def train_step(model, optimizer, t, Y_bus_tf, bus_data, attack_actions, defend_actions):
    """Train step that takes in both attack and defend actions."""
    with tf.GradientTape() as tape:
        # Compute the physics loss using both attack and defend actions
        total_loss, power_flow_loss, evcs_total_loss, wac_loss, V_regulation_loss = physics_loss(
            model, t, Y_bus_tf, bus_data, attack_actions, defend_actions
        )

    # Compute gradients and apply to model variables
    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, power_flow_loss, evcs_total_loss, wac_loss, V_regulation_loss

def train_model(initial_model=None, dqn_agent=None, sac_attacker=None, sac_defender=None, epochs=2500, batch_size=64):
    """Modified training function to handle three agents: DQN, SAC Attacker, and SAC Defender."""
    # Initialize the PINN model if not provided
    if initial_model is None:
        model = EVCS_PowerSystem_PINN()
    else:
        model = initial_model

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # Ensure the model is built by calling it once with a dummy input
    dummy_input = tf.random.uniform((1, 1))
    _ = model(dummy_input)

    # Use the custom environment for training with both agents
    env = CompetingHybridEnv(
        pinn_model=model,
        y_bus_tf=Y_bus_tf,
        bus_data=bus_data,
        v_base_lv=V_BASE_DC,
        dqn_agent=dqn_agent,
        sac_attacker=sac_attacker,
        sac_defender=sac_defender,
        num_evcs=NUM_EVCS,
        num_buses=NUM_BUSES,
        time_step=TIME_STEP
    )

    for epoch in range(epochs):
        try:
            # Use RL agents to get optimal actions for both attack and defense at the beginning of each epoch
            if dqn_agent is not None and sac_attacker is not None and sac_defender is not None:
                state = env.reset()[0]  # Extract only the observation from the reset tuple

                # Step 1: Get discrete decisions from DQN (target and duration)
                dqn_action, _ = dqn_agent.predict(state)
                

                # Step 2: Use SAC Attacker for fine-grained value selection based on DQN output
                sac_attacker_action, _ = sac_attacker.predict(state)

                # Step 3: Use SAC Defender for determining control adjustments based on current state
                sac_defender_action, _ = sac_defender.predict(state)

                # Step 4: Combine the actions from DQN and SAC agents
                attack_actions = tf.constant(sac_attacker_action, dtype=tf.float32)  # Continuous attack values
                defend_actions = tf.constant(sac_defender_action, dtype=tf.float32)  # Control adjustments

                # Optionally, incorporate the DQN action into the attack actions if needed
                # For example, you might modify the attack actions based on the DQN's selected targets

            else:
                # Use default values if no RL agents are provided
                attack_actions = tf.constant([0.0] * (NUM_EVCS * 2), dtype=tf.float32)
                defend_actions = tf.constant([0.0] * (NUM_EVCS * 2), dtype=tf.float32)

            # Run training step with updated actions for both attack and defense
            t_batch = tf.random.uniform((batch_size, 1), minval=0, maxval=TOTAL_TIME)
            total_loss, power_flow_loss, evcs_total_loss, wac_loss, V_regulation_loss = train_step(
                model, optimizer, t_batch, Y_bus_tf, tf.constant(bus_data, dtype=tf.float32),
                tf.tile(tf.expand_dims(attack_actions, 0), [batch_size, 1]),
                tf.tile(tf.expand_dims(defend_actions, 0), [batch_size, 1])
            )

            # Print loss metrics every 500 epochs
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Total Loss: {total_loss.numpy():.6f}, Power Flow Loss: {power_flow_loss.numpy():.6f}, EVCS Loss: {evcs_total_loss.numpy():.6f}")

        except Exception as e:
            print(f"Error during epoch {epoch}: {str(e)}")
            break

    return model

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
        return None


def plot_evaluation_results(results):
    """Plot the evaluation results."""
    try:
        # Convert data to numpy arrays if they aren't already
        time_steps = np.array(results['time_steps'])
        cumulative_deviations = np.array(results['cumulative_deviations'])
        voltage_deviations = np.array(results['voltage_deviations'])
        attack_active_states = np.array(results['attack_active_states'])
        target_evcs_history = np.array(results['target_evcs_history'])
        
        # Create figure with subplots
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Cumulative Voltage Deviation
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, cumulative_deviations, 'b-', label='Cumulative Deviation')
        plt.axhline(y=80, color='r', linestyle='--', label='Circuit Breaker Threshold')
        
        # Highlight attack periods
        attack_mask = attack_active_states.astype(bool)
        if np.any(attack_mask):
            plt.fill_between(time_steps, 0, np.max(cumulative_deviations),
                           where=attack_mask, color='red', alpha=0.2,
                           label='Attack Active')
        
        plt.title('System Performance During Attacks')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Cumulative Deviation')
        plt.legend()
        plt.grid(True)

        # Plot 2: Individual EVCS Voltage Deviations
        plt.subplot(3, 1, 2)
        for i in range(len(voltage_deviations[0])):  # Number of EVCSs
            plt.plot(time_steps, voltage_deviations[:, i], 
                    label=f'EVCS {i+1}')
        plt.title('Individual EVCS Voltage Deviations')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Voltage Deviation')
        plt.legend()
        plt.grid(True)

        # Plot 3: Target EVCS Selection
        plt.subplot(3, 1, 3)
        for i in range(len(target_evcs_history[0])):  # Number of EVCSs
            plt.plot(time_steps, target_evcs_history[:, i],
                    label=f'EVCS {i+1}', marker='.', markersize=2)
        plt.title('Target EVCS Selection')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Targeted (1) / Not Targeted (0)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Plot Average Attack Duration Bar Plot
        plt.figure(figsize=(10, 6))
        avg_attack_durations = np.array(results['avg_attack_durations'])
        bars = plt.bar(range(1, len(avg_attack_durations) + 1), avg_attack_durations)
        plt.title('Average Attack Duration per EVCS')
        plt.xlabel('EVCS Number')
        plt.ylabel('Average Duration (seconds)')
        plt.xticks(range(1, len(avg_attack_durations) + 1))
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_attack_durations):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, value,
                        f'{value:.2f}s',
                        ha='center', va='bottom')
            else:
                plt.text(bar.get_x() + bar.get_width()/2, 0,
                        'No attacks',
                        ha='center', va='bottom')
        
        plt.grid(True)
        plt.show()

        # Print attack statistics
        print("\nAttack Statistics:")
        for i in range(len(avg_attack_durations)):
            print(f"\nEVCS {i+1}:")
            print(f"Number of attacks: {results['attack_counts'][i]}")
            print(f"Total duration: {results['total_durations'][i]}s")
            print(f"Average duration: {avg_attack_durations[i]:.2f}s")

    except Exception as e:
        print(f"Error in plotting results: {e}")
        traceback.print_exc()

def validate_physics_constraints(env, dqn_agent, sac_attacker, sac_defender, num_episodes=5):
    """Validate that the agents respect physics constraints."""
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 100:  # Add step limit to prevent infinite loops
            try:
                # Get DQN action and decode it
                dqn_action_scalar = dqn_agent.predict(state, deterministic=True)[0]
                dqn_action = env.decode_dqn_action(dqn_action_scalar)
                
                # Get SAC actions
                attacker_action = sac_attacker.predict(state, deterministic=True)[0]
                defender_action = sac_defender.predict(state, deterministic=True)[0]
                
                # Ensure actions are properly shaped numpy arrays
                attacker_action = np.array(attacker_action).reshape(-1)
                defender_action = np.array(defender_action).reshape(-1)
                
                # Combine actions into dictionary
                action = {
                    'dqn': dqn_action,
                    'attacker': attacker_action,
                    'defender': defender_action
                }
                
                # Take step in environment
                next_state, rewards, done, truncated, info = env.step(action)
                
                # Validate physics constraints
                if isinstance(next_state, (list, tuple)):
                    next_state = np.array(next_state)
                
                if not env.validate_physics(next_state):
                    print(f"Physics constraint violation in episode {episode}, step {step_count}")
                    print(f"State values: {next_state}")
                    return False
                
                state = next_state
                step_count += 1
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                return False
            
    return True


if __name__ == '__main__':
    # Define physics parameters
    physics_params = {
        'voltage_limits': (0.9, 1.1),
        'v_out_nominal': 1.0,
        'current_limits': (-1.0, 1.0),
        'i_rated': 1.0,
        'attack_magnitude': 0.05,
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
        exploration_fraction=1,
        exploration_final_eps=0.1,
        train_freq=4,
        batch_size=32,
        gamma=0.99,
        device='cuda',
        tensorboard_log=f"{log_dir}/dqn/"
    )

    # Train DQN with monitoring
    print("Training DQN agent...")
    dqn_agent.learn(
        total_timesteps=10000,
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

    # ... (previous code remains the same until combined_env creation)

    # Initialize SAC Wrapper environments
    # Create SAC Wrapper environments with proper initialization
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
        learning_rate=5e-3,
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
        learning_rate=1e-5,
        buffer_size=5000,
        batch_size=64,
        gamma=0.95,
        tau=0.01,
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
        total_timesteps=10000,
        callback=sac_attacker_checkpoint,
        progress_bar=True
    )
    sac_attacker.save(f"{model_dir}/sac_attacker_final")

    print("Training the SAC Defender agent...")
    sac_defender.learn(
        total_timesteps=10000,
        callback=sac_defender_checkpoint,
        progress_bar=True
    )
    sac_defender.save(f"{model_dir}/sac_defender_final")

    num_iterations = 5
    # # Joint training loop with validation
    # print("Starting joint training...")
    # for iteration in range(num_iterations):
    #     print(f"\nJoint training iteration {iteration + 1}/{num_iterations}")
        
    #     # Train agents with progress monitoring
    #     for agent, name, callback, env in [
    #         (dqn_agent, "DQN", dqn_checkpoint, discrete_env),
    #         (sac_attacker, "SAC Attacker", sac_attacker_checkpoint, sac_attacker_env),
    #         (sac_defender, "SAC Defender", sac_defender_checkpoint, sac_defender_env)
    #     ]:
    #         print(f"\nTraining {name}...")
    #         agent.learn(
    #             total_timesteps=5000,
    #             callback=callback,
    #             progress_bar=True
    #         )
    #         agent.save(f"{model_dir}/{name.lower()}_iter_{iteration + 1}")

    #         # Update environment references after each agent training
    #         combined_env.update_agents(dqn_agent, sac_attacker, sac_defender)
    #         sac_attacker_env.update_agents(sac_defender=sac_defender, dqn_agent=dqn_agent)
    #         sac_defender_env.update_agents(sac_attacker=sac_attacker, dqn_agent=dqn_agent)

    print("Starting joint training...")
    for iteration in range(num_iterations):
        print(f"\nJoint training iteration {iteration + 1}/{num_iterations}")
        
        # Train agents with different timesteps
        training_config = [
            (dqn_agent, "DQN", dqn_checkpoint, discrete_env, 15000),
            (sac_attacker, "SAC Attacker", sac_attacker_checkpoint, sac_attacker_env, 15000),
            (sac_defender, "SAC Defender", sac_defender_checkpoint, sac_defender_env, 10000)  # Reduced timesteps
        ]
        
        for agent, name, callback, env, timesteps in training_config:
            print(f"\nTraining {name}...")
            agent.learn(
                total_timesteps=timesteps,
                callback=callback,
                progress_bar=True
            )
            agent.save(f"{model_dir}/{name.lower()}_iter_{iteration + 1}")
            
            # Update environment references
            combined_env.update_agents(dqn_agent, sac_attacker, sac_defender)
            sac_attacker_env.update_agents(sac_defender=sac_defender, dqn_agent=dqn_agent)
            sac_defender_env.update_agents(sac_attacker=sac_attacker, dqn_agent=dqn_agent)

            # Validate physics constraints
            print("\nValidating physics constraints...")
            validation_success = validate_physics_constraints(
                combined_env,
                dqn_agent,
                sac_attacker,
                sac_defender,
                num_episodes=5
            )
            print(f"Physics validation: {'Passed' if validation_success else 'Failed'}")


    print("Training the PINN model with the hybrid RL agents (DQN for target, SAC Attacker for FDI, and SAC Defender for stabilization)...")
    trained_pinn_model = train_model(
    initial_model=initial_pinn_model,
    dqn_agent=dqn_agent,
    sac_attacker=sac_attacker,
    sac_defender=sac_defender,
    epochs=7000,
    batch_size=128
    )

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
        'voltage_deviations': results['voltage_deviations'].tolist(),
        'attack_active_states': results['attack_active_states'].tolist(),
        'target_evcs_history': [targets.tolist() if isinstance(targets, np.ndarray) else targets 
                               for targets in results['target_evcs_history']],
        'attack_durations': results['attack_durations'].tolist(),
        'avg_attack_durations': results['avg_attack_durations'].tolist(),
        'attack_counts': results['attack_counts'],
        'total_durations': results['total_durations']
    }

    # Save and plot evaluation results
    if results is not None:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'time_steps': results['time_steps'].tolist(),
            'cumulative_deviations': results['cumulative_deviations'].tolist(),
            'voltage_deviations': results['voltage_deviations'].tolist(),
            'attack_active_states': results['attack_active_states'].tolist(),
            'target_evcs_history': [targets.tolist() if isinstance(targets, np.ndarray) else targets 
                                   for targets in results['target_evcs_history']],
            'attack_durations': results['attack_durations'].tolist(),
            'avg_attack_durations': results['avg_attack_durations'].tolist(),
            'attack_counts': results['attack_counts'],
            'total_durations': results['total_durations']
        }

        # Save results
        with open(f"{log_dir}/final_evaluation_results.json", "w") as f:
            json.dump(serializable_results, f, indent=4)

        # Plot results
        plot_evaluation_results(serializable_results)
    else:
        print("Error: No results available to plot")

    print("\nTraining completed successfully!")

