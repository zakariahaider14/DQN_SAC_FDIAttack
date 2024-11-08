import gymnasium as gym
import numpy as np
import tensorflow as tf
from stable_baselines3 import SAC, DQN
import matplotlib.pyplot as plt

# Environment for DQN Agent (Discrete Action Space)
class DiscreteHybridEnv(gym.Env):
    def __init__(self, pinn_model, y_bus_tf, bus_data, v_base_lv, num_evcs=5, num_buses=33, time_step=0.1, **physics_params):
        super(DiscreteHybridEnv, self).__init__()
        self.pinn_model = pinn_model
        self.Y_bus_tf = y_bus_tf
        self.bus_data = bus_data
        self.V_BASE_LV = v_base_lv
        self.NUM_EVCS = num_evcs
        self.NUM_BUSES = num_buses
        self.TIME_STEP = time_step
        
        # Define action space parameters
        self.NUM_ACTIONS = 2  # binary actions for each EVCS (0 or 1)
        self.NUM_DURATION = 10  # number of possible duration values
        
        # Calculate total number of possible actions
        total_actions = (2 ** self.NUM_EVCS) * self.NUM_DURATION
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(total_actions)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(25,),  # Adjust based on your state space
            dtype=np.float32
        )
        
        # Initialize other attributes
        self.time_step_counter = 0
        self.attack_start_time = 0
        self.attack_end_time = 0
        self.attack_active = False
        self.current_targets = np.zeros(self.NUM_EVCS, dtype=np.int32)
        self.current_duration = 0
        
        # Extract physics parameters with defaults
        self.voltage_limits = physics_params.get('voltage_limits', (0.85, 1.15))
        self.v_out_nominal = physics_params.get('v_out_nominal', 1.0)
        self.current_limits = physics_params.get('current_limits', (-1.0, 1.0))
        self.i_rated = physics_params.get('i_rated', 1.0)
        # self.attack_magnitude = physics_params.get('attack_magnitude', 0.01)
        # self.current_magnitude = physics_params.get('current_magnitude', 0.01)
        self.wac_kp_limits = physics_params.get('wac_kp_limits', (0.0, 2.0))
        self.wac_ki_limits = physics_params.get('wac_ki_limits', (0.0, 2.0))
        self.control_saturation = physics_params.get('control_saturation', 0.3)
        self.power_limits = physics_params.get('power_limits', (-1.0, 1.0))
        self.power_ramp_rate = physics_params.get('power_ramp_rate', 0.1)
        self.evcs_efficiency = physics_params.get('evcs_efficiency', 0.98)
        self.soc_limits = physics_params.get('soc_limits', (0.1, 0.9))
        self.voltage_deviations = np.zeros(self.NUM_EVCS)


        # self.attack_magnitude = np.random.uniform(0.005, 0.01)
        # self.current_magnitude = np.random.uniform(0.005, 0.01)
        
        # Initialize attack-related parameters
        self.attack_duration = 0
        self.target_evcs = [0] * self.NUM_EVCS
        self.attack_active = False
        self.attack_start_time = 0
        self.attack_end_time = 0

        # Discrete Action Space for DQN
        self.dqn_action_space = gym.spaces.MultiDiscrete([2] * self.NUM_EVCS + [10])
        total_dqn_actions = int(np.prod([2] * self.NUM_EVCS + [10]))
        self.action_space = gym.spaces.Discrete(total_dqn_actions)

        # Observation Space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        self.state = np.zeros(self.observation_space.shape[0])
        
        self.reset_state()

    def reset_state(self):
        """Helper function to reset internal state variables."""
        self.state = np.zeros(self.observation_space.shape[0])
        self.current_time = 0.0
        self.time_step_counter = 0
        self.cumulative_deviation = 0.0
        self.attack_duration = 0
        self.target_evcs = [0] * self.NUM_EVCS
        self.attack_active = False
        self.attack_start_time = 0
        self.attack_end_time = 0
        self.voltage_deviations = np.zeros(self.NUM_EVCS)

    def reset(self, **kwargs):
        """Reset the environment and get the initial state."""
        self.reset_state()
        initial_prediction = self.pinn_model(tf.constant([[self.current_time]], dtype=tf.float32))
        evcs_vars = initial_prediction[:, 2 * self.NUM_BUSES:].numpy()[0]
        self.state = self.get_observation(evcs_vars)
        return self.state, {}

    def calculate_attack_duration(self, decoded_action):
        """Calculate the actual attack duration in time steps."""
        duration_value = decoded_action[-1]
        # Map duration_value (0 to NUM_DURATION-1) to actual time steps
        # For example, if each increment represents 40 time steps
        return duration_value * 40  # This will give durations from 0 to 360 time steps

    def apply_attack_effect(self, i):
        """Apply attack effects with random magnitudes within specified ranges."""
        if self.target_evcs[i] == 1:
            # Generate random attack magnitudes
            random_voltage_magnitude = np.random.uniform(0.0005, 0.001)
            random_current_magnitude = np.random.uniform(0.0005, 0.001)
            
            # Create temporary state for validation
            temp_state = self.state.copy()
            
            # Apply voltage and current deviations
            temp_state[i] = self.state[i] + random_voltage_magnitude
            temp_state[15 + i] = self.state[15 + i] + random_current_magnitude
            
            # Validate physics constraints before applying
            if self.validate_physics(temp_state):
                self.state = temp_state
                # print(f"Attack applied to EVCS {i}: Voltage +{random_voltage_magnitude:.3f}, Current +{random_current_magnitude:.3f}")
            else:
                print(f"Attack on EVCS {i} rejected: Physics constraints violated")
        
        return self.state

    def validate_physics(self, state):
        """Validate that state updates comply with physical constraints"""
        # Validate voltages
        v_out = state[:self.NUM_EVCS]
        if not np.all((v_out >= self.voltage_limits[0]) & (v_out <= self.voltage_limits[1])):
            return False
            
        # Validate currents
        i_out = state[15:15+self.NUM_EVCS]
        i_dc = state[20:20+self.NUM_EVCS]
        if not np.all((i_out >= self.current_limits[0]) & (i_out<= self.current_limits[1])):
            return False
        if not np.all((i_dc >= self.current_limits[0]) & (i_dc <= self.current_limits[1])):
            return False
            
        return True

    def step(self, encoded_dqn_action):
        """Execute one time step within the environment."""
        # Decode the action and convert to numpy array to ensure proper handling
        dqn_action = np.array(self.decode_action(encoded_dqn_action))
        
        # Set attack parameters
        self.attack_active = np.any(dqn_action[:-1] > 0)  # Check if any EVCS is targeted
        self.attack_start_time = self.time_step_counter
        self.attack_duration = self.calculate_attack_duration(dqn_action)
        self.target_evcs = dqn_action[:-1].astype(int)  # Convert to list for consistency
        self.attack_end_time = self.attack_start_time + self.attack_duration    # Get PINN prediction for the current time step

        
        # print(f"Encoded action: {encoded_dqn_action}")
        # print(f"Decoded action: {dqn_action}")
        # print(f"Target EVCSs in DQN: {self.target_evcs}")
        # print(f"Attack duration: {self.attack_duration}")
        # print(f"Attack active: {self.attack_active}")
        # print(f"Attack start time: {self.attack_start_time}")
        # print(f"Attack end time: {self.attack_end_time}")


        prediction = self.pinn_model(tf.constant([[self.current_time]], dtype=tf.float32))
        evcs_vars = prediction[:, 2 * self.NUM_BUSES:].numpy()[0]
        new_state = self.get_observation(evcs_vars)

        # Validate initial state update from PINN
        if self.validate_physics(new_state):
            self.state = new_state.copy()  # Use copy to prevent reference issues
        else:
            print("Warning: PINN prediction violated physics constraints")
            # Optionally clip values to valid ranges instead of rejecting
            self.state = np.clip(new_state, 
                               [self.voltage_limits[0]] * self.NUM_EVCS + [-np.inf] * 20,
                               [self.voltage_limits[1]] * self.NUM_EVCS + [np.inf] * 20)

        # Apply attack effects if active
        if self.attack_active and self.time_step_counter <= self.attack_end_time:
            for i in range(self.NUM_EVCS):
                if self.target_evcs[i] == 1:
                    self.state = self.apply_attack_effect(i)

        # Validate and update state
  
        # Update timing
        self.current_time += self.TIME_STEP
        self.time_step_counter += 1
        # print("Time step counter: ", self.time_step_counter)

        # Calculate rewards
        self.voltage_deviations = np.abs(self.state[:self.NUM_EVCS] - 1.0)
        max_deviations = np.max(self.voltage_deviations )
        rewards = self.calculate_rewards(self.voltage_deviations)
        total_reward = sum(rewards)

        # Check termination conditions
        # done = self.time_step_counter >= 1000 or np.any( voltage_deviations > 0.5)

        done = self.time_step_counter >= 1000 or max_deviations >= 0.4
        truncated = False

        info = {
            'voltage_deviations': self.voltage_deviations,
            'individual_rewards': rewards,
            'time_step': self.time_step_counter,
            'attack_active': self.attack_active,
            'attack_duration': self.attack_duration,
            'total_reward': total_reward
        }

        return self.state, total_reward, done, truncated, info

    def calculate_rewards(self, voltage_deviations):
        """Calculate rewards based on voltage deviations."""
        rewards = []
        for i, deviation in enumerate(voltage_deviations):
            if deviation > 0.5:
                # Higher reward for successful attack (larger deviation)
                rewards.append(100 - 0.1 * self.current_time)
            else:
                # Lower reward for unsuccessful attack (smaller deviation)
                rewards.append(-1* self.current_time - deviation)
        return rewards

    def decode_action(self, action_scalar):
        """Decode a scalar action into target EVCSs and duration."""
        try:
            # Ensure action_scalar is an integer
            if isinstance(action_scalar, (np.ndarray, np.generic)):
                action_scalar = int(action_scalar.item())
            elif not isinstance(action_scalar, (int, np.integer)):
                raise ValueError(f"Expected integer action, got {type(action_scalar)}")

            # Validate action range
            if action_scalar >= self.action_space.n:
                raise ValueError(f"Action {action_scalar} exceeds action space size {self.action_space.n}")

            # Calculate target value and duration
            target_value = action_scalar // self.NUM_DURATION
            duration_value = action_scalar % self.NUM_DURATION

            # Convert target value to binary array
            target_evcs = np.zeros(self.NUM_EVCS, dtype=np.int32)
            for i in range(self.NUM_EVCS):
                target_evcs[i] = (target_value >> i) & 1

            # Combine into final action array
            decoded_action = np.append(target_evcs, duration_value)

            # # Debug information
            # print(f"Action scalar: {action_scalar}")
            # print(f"Target value: {target_value}")
            # print(f"Duration value: {duration_value}")
            # print(f"Target EVCSs: {target_evcs}")
            # print(f"Duration: {duration_value}")

            return decoded_action

        except Exception as e:
            print(f"Error decoding action {action_scalar}: {str(e)}")
            # Return safe default action
            return np.zeros(self.NUM_EVCS + 1, dtype=np.int32)

    def get_observation(self, evcs_vars):
        """Convert EVCS variables into observation format."""
        v_out_values, soc_values, v_dc_values, i_out_values, i_dc_values = [], [], [], [], []
        for i in range(self.NUM_EVCS):
            v_dc = np.exp(evcs_vars[i * 18 + 2])
            v_out = np.exp(evcs_vars[i * 18 + 4])
            soc = evcs_vars[i * 18 + 9]
            i_out = evcs_vars[i * 18 + 16]
            i_dc = evcs_vars[i * 18 + 17]
            v_out_values.append(v_out)
            soc_values.append(soc)
            v_dc_values.append(v_dc)
            i_out_values.append(i_out)
            i_dc_values.append(i_dc)
        return np.concatenate([v_out_values, soc_values, v_dc_values, i_out_values, i_dc_values])
