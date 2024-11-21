import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, DQN
import matplotlib.pyplot as plt

class DiscreteHybridEnv(gym.Env):
    def __init__(self, pinn_model, y_bus_torch, bus_data, v_base_lv, num_evcs=5, num_buses=33, time_step=0.1, **physics_params):
        super(DiscreteHybridEnv, self).__init__()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models and data
        self.pinn_model = pinn_model
        self.Y_bus_torch = y_bus_torch
        self.bus_data = bus_data
        self.V_BASE_LV = torch.tensor(v_base_lv, device=self.device)
        self.NUM_EVCS = num_evcs
        self.NUM_BUSES = num_buses
        self.TIME_STEP = time_step
        
        # Define action space parameters
        self.NUM_ACTIONS = 2  # binary actions for each EVCS (0 or 1)
        self.NUM_DURATION = 10  # number of possible duration values
        
        # Calculate total number of possible actions
        total_actions = (2 ** self.NUM_EVCS) * self.NUM_DURATION
        
        # Define action and observation spaces (keep as numpy for gym compatibility)
        self.action_space = gym.spaces.Discrete(total_actions)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(25,),
            dtype=np.float32
        )
        
        # Initialize state variables as tensors
        self.time_step_counter = 0
        self.attack_start_time = torch.tensor(0, device=self.device)
        self.attack_end_time = torch.tensor(0, device=self.device)
        self.attack_active = False
        self.current_targets = torch.zeros(self.NUM_EVCS, dtype=torch.int64, device=self.device)
        self.current_duration = torch.tensor(0, device=self.device)
        
        # Extract physics parameters with defaults
        self.voltage_limits = (torch.tensor(physics_params.get('voltage_limits', (0.85, 1.15))[0], device=self.device),
                             torch.tensor(physics_params.get('voltage_limits', (0.85, 1.15))[1], device=self.device))
        self.v_out_nominal = torch.tensor(physics_params.get('v_out_nominal', 1.0), device=self.device)
        self.current_limits = (torch.tensor(physics_params.get('current_limits', (-1.0, 1.0))[0], device=self.device),
                             torch.tensor(physics_params.get('current_limits', (-1.0, 1.0))[1], device=self.device))
        self.i_rated = torch.tensor(physics_params.get('i_rated', 1.0), device=self.device)
        self.wac_kp_limits = physics_params.get('wac_kp_limits', (0.0, 2.0))
        self.wac_ki_limits = physics_params.get('wac_ki_limits', (0.0, 2.0))
        self.control_saturation = torch.tensor(physics_params.get('control_saturation', 0.3), device=self.device)
        self.power_limits = physics_params.get('power_limits', (-1.0, 1.0))
        self.power_ramp_rate = torch.tensor(physics_params.get('power_ramp_rate', 0.1), device=self.device)
        self.evcs_efficiency = torch.tensor(physics_params.get('evcs_efficiency', 0.98), device=self.device)
        self.soc_limits = physics_params.get('soc_limits', (0.1, 0.9))
        self.voltage_deviations = torch.zeros(self.NUM_EVCS, device=self.device)
        
        # Initialize attack-related parameters
        self.attack_duration = torch.tensor(0, device=self.device)
        self.target_evcs = torch.zeros(self.NUM_EVCS, dtype=torch.int64, device=self.device)
        
        # Discrete Action Space for DQN (keep as numpy for gym compatibility)
        self.dqn_action_space = gym.spaces.MultiDiscrete([2] * self.NUM_EVCS + [10])
        total_dqn_actions = int(torch.prod(torch.tensor([2] * self.NUM_EVCS + [10])))
        self.action_space = gym.spaces.Discrete(total_dqn_actions)
        
        # Initialize state tensor
        self.state = torch.zeros(self.observation_space.shape[0], device=self.device)
        
        self.reset_state()

    def reset_state(self):
        """Helper function to reset internal state variables."""
        self.state = torch.zeros(self.observation_space.shape[0], device=self.device)
        self.current_time = torch.tensor(0.0, device=self.device)
        self.time_step_counter = 0
        self.cumulative_deviation = torch.tensor(0.0, device=self.device)
        self.attack_duration = torch.tensor(0, device=self.device)
        self.target_evcs = torch.zeros(self.NUM_EVCS, dtype=torch.int64, device=self.device)
        self.attack_active = False
        self.attack_start_time = torch.tensor(0, device=self.device)
        self.attack_end_time = torch.tensor(0, device=self.device)
        self.voltage_deviations = torch.zeros(self.NUM_EVCS, device=self.device)

    def reset(self, seed=None, options=None):
        """Reset the environment and get the initial state."""
        self.reset_state()
        initial_time = torch.tensor([[self.current_time]], device=self.device)
        initial_prediction = self.pinn_model(initial_time)
        evcs_vars = initial_prediction[:, 2 * self.NUM_BUSES:].detach().cpu().numpy()[0]
        self.state = self.get_observation(evcs_vars)
        return self.state.cpu().numpy(), {}

    def validate_physics(self, state):
        """Validate that state updates comply with physical constraints"""
        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self.device)
            
        # Validate voltages
        v_out = state[:self.NUM_EVCS]
        if not torch.all((v_out >= self.voltage_limits[0]) & (v_out <= self.voltage_limits[1])):
            return False
            
        # Validate currents
        i_out = state[15:15+self.NUM_EVCS]
        i_dc = state[20:20+self.NUM_EVCS]
        if not torch.all((i_out >= self.current_limits[0]) & (i_out <= self.current_limits[1])):
            return False
        if not torch.all((i_dc >= self.current_limits[0]) & (i_dc <= self.current_limits[1])):
            return False
            
        return True

    def apply_attack_effect(self, i):
        """Apply attack effects with random magnitudes within specified ranges."""
        if self.target_evcs[i] == 1:
            # Generate random attack magnitudes
            random_voltage_magnitude = torch.rand(1, device=self.device) * 0.0005 + 0.0005
            random_current_magnitude = torch.rand(1, device=self.device) * 0.0005 + 0.0005
            
            # Create temporary state for validation
            temp_state = self.state.clone()
            
            # Apply voltage and current deviations
            temp_state[i] = self.state[i] + random_voltage_magnitude
            temp_state[15 + i] = self.state[15 + i] + random_current_magnitude
            
            # Validate physics constraints before applying
            if self.validate_physics(temp_state):
                self.state = temp_state
            else:
                print(f"Attack on EVCS {i} rejected: Physics constraints violated")
        
        return self.state
    

    def calculate_attack_duration(self, decoded_action):
        """Calculate the actual attack duration in time steps."""
        duration_value = decoded_action[-1]
        # Define base duration and scaling factor
        BASE_DURATION = 10  # minimum duration
        DURATION_SCALE = 10  # how much each increment adds
        
        # Calculate duration: BASE_DURATION + (value * DURATION_SCALE)
        return BASE_DURATION + (duration_value * DURATION_SCALE)  # gives durations from 10 to 100 steps

    def step(self, action):
        try:
            with torch.no_grad():  # Prevent gradient tracking
                # Decode action
                dqn_action = self.decode_action(action)
                
                # Get PINN prediction
                current_time = torch.tensor([[self.current_time]], device=self.device)
                prediction = self.pinn_model(current_time)
                evcs_vars = prediction[:, 2 * self.NUM_BUSES:].detach().cpu().numpy()[0]
                new_state = self.get_observation(evcs_vars)

                # Set attack parameters
                self.attack_active = torch.any(dqn_action[:-1] > 0).detach()
                self.target_evcs = dqn_action[:-1].to(torch.int64).detach()
                self.attack_duration = self.calculate_attack_duration(dqn_action).detach()
                
                # Validate and update state
                if self.validate_physics(new_state):
                    self.state = new_state.detach().clone()
                else:
                    self.state = torch.clamp(
                        new_state.detach(),
                        torch.cat([
                            torch.full((self.NUM_EVCS,), self.voltage_limits[0], device=self.device),
                            torch.full((20,), float('-inf'), device=self.device)
                        ]),
                        torch.cat([
                            torch.full((self.NUM_EVCS,), self.voltage_limits[1], device=self.device),
                            torch.full((20,), float('inf'), device=self.device)
                        ])
                    ).detach()

                # Apply attack effects
                if self.attack_active and self.time_step_counter <= self.attack_end_time:
                    for i in range(self.NUM_EVCS):
                        if self.target_evcs[i] == 1:
                            self.state = self.apply_attack_effect(i).detach()

                # Calculate rewards and deviations
                self.voltage_deviations = torch.abs(self.state[:self.NUM_EVCS] - 1.0).detach()
                max_deviations = torch.max(self.voltage_deviations).detach()
                rewards = self.calculate_rewards(self.voltage_deviations)
                truncated = False
                
                # Prepare return values
                state = self.state.detach().cpu().numpy()
                rewards = rewards.detach().cpu().numpy() if torch.is_tensor(rewards) else rewards

                done = self.time_step_counter >= 1000 or max_deviations >= 0.5
                
                info = {
                    'voltage_deviations': self.voltage_deviations.detach().cpu().numpy(),
                    'individual_rewards': rewards,
                    'time_step': self.time_step_counter,
                    'attack_active': self.attack_active.item(),
                    'attack_duration': self.attack_duration.item(),
                    'total_reward': sum(rewards)
                }
                
                # Convert list rewards to single float for DQN
                if isinstance(rewards, list):
                    rewards = float(sum(rewards))  # Sum if it's a list
                elif isinstance(rewards, dict):
                    rewards = float(sum(rewards.values()))  # Sum if it's a dict
                
                return state, rewards, done, truncated, info

        except Exception as e:
            print(f"Error in step: {str(e)}")
            return self.state.detach().cpu().numpy(), 0, True, False, {}

    def calculate_rewards(self, voltage_deviations):
        """Calculate rewards based on voltage deviations."""
        rewards = []
        for i, deviation in enumerate(voltage_deviations):
            if deviation > 0.1:
                # Higher reward for successful attack
                rewards.append(100 - 0.1 * self.current_time)
            else:
                # Lower reward for unsuccessful attack
                rewards.append(-1 * self.current_time - deviation.item())
        return rewards

    def decode_action(self, action_scalar):
        """Decode a scalar action into target EVCSs and duration."""
        try:
            # Convert to tensor if needed
            if not torch.is_tensor(action_scalar):
                action_scalar = torch.tensor(action_scalar, device=self.device)
            
            # Convert to scalar if needed
            if action_scalar.numel() > 1:
                action_scalar = action_scalar.item()
            
            # Validate action range
            action_scalar = int(action_scalar)
            if action_scalar >= self.action_space.n:
                raise ValueError(f"Action {action_scalar} exceeds action space size {self.action_space.n}")

            # Calculate target value and duration
            target_value = action_scalar // self.NUM_DURATION
            duration_value = action_scalar % self.NUM_DURATION

            # Convert target value to binary tensor
            target_evcs = torch.zeros(self.NUM_EVCS, dtype=torch.int64, device=self.device)
            for i in range(self.NUM_EVCS):
                target_evcs[i] = (target_value >> i) & 1

            # Return decoded action
            return torch.cat([target_evcs, torch.tensor([duration_value], device=self.device)])

        except Exception as e:
            print(f"Error decoding action {action_scalar}: {str(e)}")
            return torch.zeros(self.NUM_EVCS + 1, dtype=torch.int64, device=self.device)

    def get_observation(self, evcs_vars):
        """Convert EVCS variables into observation format."""
        if not torch.is_tensor(evcs_vars):
            evcs_vars = torch.tensor(evcs_vars, device=self.device)
            
        v_out_values = []
        soc_values = []
        v_dc_values = []
        i_out_values = []
        i_dc_values = []
        
        for i in range(self.NUM_EVCS):
            v_dc = torch.exp(evcs_vars[i * 18 + 2])
            v_out = torch.exp(evcs_vars[i * 18 + 4])
            soc = evcs_vars[i * 18 + 9]
            i_out = evcs_vars[i * 18 + 16]
            i_dc = evcs_vars[i * 18 + 17]
            
            v_out_values.append(v_out)
            soc_values.append(soc)
            v_dc_values.append(v_dc)
            i_out_values.append(i_out)
            i_dc_values.append(i_dc)
        
        return torch.cat([
            torch.stack(v_out_values),
            torch.stack(soc_values),
            torch.stack(v_dc_values),
            torch.stack(i_out_values),
            torch.stack(i_dc_values)
        ])

    def get_pinn_state(self):
        """Get state from PINN model."""
        try:
            # Create time input for PINN model
            t = torch.tensor([[self.current_time]], device=self.device)
            
            # Get PINN model predictions
            pinn_outputs = self.pinn_model(t)
            
            # Extract voltage and EVCS variables
            num_voltage_outputs = self.NUM_BUSES * 2
            voltage_outputs = pinn_outputs[:, :num_voltage_outputs]
            evcs_vars = pinn_outputs[:, num_voltage_outputs:]
            
            # Convert to observation format
            observation = self.get_observation(evcs_vars[0])
            return observation
            
        except Exception as e:
            print(f"Error in get_pinn_state: {e}")
            return torch.zeros(self.observation_space.shape[0], device=self.device)

    def decode_dqn_action(self, action):
        """Decode DQN action into environment action format"""
        return int(action)  # or your specific decoding logic

    def reset(self, seed=None, options=None):
        """Reset environment with seed support"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Reset environment state
        self.state = torch.zeros(self.observation_space.shape[0], dtype=torch.float32)
        self.time_step_counter = 0
        self.attack_duration = 0
        
        return self.state.detach().numpy(), {}  # Return state and empty info dict