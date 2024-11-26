import gymnasium as gym
import numpy as np
import torch

class CompetingHybridEnv(gym.Env):
    """Custom environment for joint training of DQN and SAC agents."""
    def __init__(self, pinn_model, y_bus_torch, bus_data, v_base_lv, dqn_agent, num_evcs=5, num_buses=33, time_step=0.1, **physics_params):
        super(CompetingHybridEnv, self).__init__()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # System parameters
        self.NUM_EVCS = num_evcs
        self.NUM_BUSES = num_buses
        self.TIME_STEP = time_step
        self.V_BASE_LV = v_base_lv

        # Extract physics parameters with defaults
        self.voltage_limits = physics_params.get('voltage_limits', (0.85, 1.15))
        self.v_out_nominal = physics_params.get('v_out_nominal', 1.0)
        self.current_limits = physics_params.get('current_limits', (-1.0, 1.0))
        self.i_rated = physics_params.get('i_rated', 1.0)
        self.attack_magnitude = physics_params.get('attack_magnitude', 0.01)
        self.current_magnitude = physics_params.get('current_magnitude', 0.01)
        self.wac_kp_limits = physics_params.get('wac_kp_limits', (0.0, 2.0))
        self.wac_ki_limits = physics_params.get('wac_ki_limits', (0.0, 2.0))
        self.control_saturation = physics_params.get('control_saturation', 0.3)
        self.power_limits = physics_params.get('power_limits', (-1.0, 1.0))
        self.power_ramp_rate = physics_params.get('power_ramp_rate', 0.1)
        self.evcs_efficiency = physics_params.get('evcs_efficiency', 0.98)
        self.soc_limits = physics_params.get('soc_limits', (0.1, 0.9))
        
        # WAC parameters
        self.WAC_VOUT_SETPOINT = torch.tensor(1.0, device=self.device)
        self.WAC_KP_VOUT_DEFAULT = torch.tensor(0.3, device=self.device)
        self.WAC_KI_VOUT_DEFAULT = torch.tensor(0.2, device=self.device)
        self.WAC_KP_VDC_DEFAULT = torch.tensor(0.3, device=self.device)
        self.WAC_KI_VDC_DEFAULT = torch.tensor(0.2, device=self.device)
        
        # Voltage limits
        self.V_OUT_NOMINAL = torch.tensor(1.0, device=self.device)
        self.V_OUT_VARIATION = torch.tensor(0.05, device=self.device)
        self.V_OUT_MIN = self.V_OUT_NOMINAL - self.V_OUT_VARIATION
        self.V_OUT_MAX = self.V_OUT_NOMINAL + self.V_OUT_VARIATION
        
        # Initialize models and data
        self.pinn_model = pinn_model.to(self.device)
        self.y_bus_torch = y_bus_torch.to(self.device)
        self.bus_data = torch.as_tensor(bus_data, device=self.device)
        self.dqn_agent = dqn_agent
        
        # Initialize state variables as tensors
        self.time_step_counter = 0
        self.current_time = torch.tensor(0.0, device=self.device)
        self.cumulative_deviation = torch.tensor(0.0, device=self.device)
        
        # Initialize attack-related variables as tensors
        self.target_evcs = torch.zeros(self.NUM_EVCS, device=self.device)
        self.attack_active = False
        self.attack_start_time = torch.tensor(0, device=self.device)
        self.attack_end_time = torch.tensor(0, device=self.device)
        self.attack_duration = torch.tensor(0, device=self.device)
        self.voltage_deviations = torch.zeros(self.NUM_EVCS, device=self.device)
        
        # WAC control variables as tensors
        self.wac_integral = torch.zeros(self.NUM_EVCS, device=self.device)
        self.wac_error = torch.zeros(self.NUM_EVCS, device=self.device)
        self.wac_control = torch.zeros(self.NUM_EVCS, device=self.device)
        self.voltage_error = torch.zeros(self.NUM_EVCS, device=self.device)
        self.kp_vout = torch.ones(self.NUM_EVCS, device=self.device) * self.WAC_KP_VOUT_DEFAULT
        self.ki_vout = torch.ones(self.NUM_EVCS, device=self.device) * self.WAC_KI_VOUT_DEFAULT
        
        # Define action spaces (keep as numpy for gym compatibility)
        self.dqn_action_space = gym.spaces.MultiDiscrete([2] * self.NUM_EVCS + [10])
        
        self.sac_attacker_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        self.sac_defender_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )

        self.state = torch.zeros(self.observation_space.shape[0], device=self.device)

        # Control limits as tensors
        self.CONTROL_MAX = torch.tensor(1.0, device=self.device)
        self.CONTROL_MIN = torch.tensor(-1.0, device=self.device)
        self.INTEGRAL_MAX = torch.tensor(10.0, device=self.device)
        self.INTEGRAL_MIN = torch.tensor(-10.0, device=self.device)

        # Add voltage and current limits as tensors
        self.voltage_limits = (torch.tensor(0.8, device=self.device), 
                             torch.tensor(1.2, device=self.device))
        self.current_limits = (torch.tensor(-1.0, device=self.device), 
                             torch.tensor(1.0, device=self.device))

        self.reset_state()

    def _setup_action_spaces(self):
        """Setup action spaces for all agents."""
        # DQN action space (keep as numpy for gym compatibility)
        self.dqn_action_space = gym.spaces.MultiDiscrete([2] * self.NUM_EVCS + [10])
        self.total_dqn_actions = int(torch.prod(torch.tensor([2] * self.NUM_EVCS + [10])))
        
        # SAC Attacker action space
        self.sac_attacker_action_space = gym.spaces.Box(
            low=-self.attack_magnitude * torch.ones(self.NUM_EVCS * 2, device=self.device).cpu().numpy(),
            high=self.attack_magnitude * torch.ones(self.NUM_EVCS * 2, device=self.device).cpu().numpy(),
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        # SAC Defender action space
        wac_limits = torch.cat([
            torch.full((self.NUM_EVCS,), self.wac_kp_limits[1], device=self.device),
            torch.full((self.NUM_EVCS,), self.wac_ki_limits[1], device=self.device)
        ])
        
        self.sac_defender_action_space = gym.spaces.Box(
            low=torch.zeros(self.NUM_EVCS * 2, device=self.device).cpu().numpy(),
            high=wac_limits.cpu().numpy(),
            shape=(self.NUM_EVCS * 2,),
            dtype=np.float32
        )
        
        # Combined action space
        self.action_space = gym.spaces.Dict({
            'dqn': self.dqn_action_space,
            'attacker': self.sac_attacker_action_space,
            'defender': self.sac_defender_action_space
        })

    def update_agents(self, dqn_agent=None, sac_attacker=None, sac_defender=None):
        """Update agent references in the environment."""
        if dqn_agent is not None:
            self.dqn_agent = dqn_agent
        if sac_attacker is not None:
            self.sac_attacker = sac_attacker
        if sac_defender is not None:
            self.sac_defender = sac_defender
        
        # Validate that required agents are present
        if self.dqn_agent is None:
            print("Warning: DQN agent is not set")
        if self.sac_attacker is None:
            print("Warning: SAC attacker is not set")
        if self.sac_defender is None:
            print("Warning: SAC defender is not set")

    def validate_agents(self):
        """Validate that all required agents are properly initialized."""
        agents_valid = True
        
        if self.dqn_agent is None:
            print("Error: DQN agent is required but not set")
            agents_valid = False
            
        if self.sac_attacker is None:
            print("Warning: SAC attacker is not set, will use default actions")
            
        if self.sac_defender is None:
            print("Warning: SAC defender is not set, will use default actions")
            
        return agents_valid

    def reset_state(self):
        """Reset all state variables."""
        self.time_step_counter = 0
        self.current_time = torch.tensor(0.0, device=self.device)
        self.cumulative_deviation = torch.tensor(0.0, device=self.device)
        
        # Reset WAC control variables
        self.wac_integral = torch.zeros(self.NUM_EVCS, device=self.device)
        self.wac_error = torch.zeros(self.NUM_EVCS, device=self.device)
        self.wac_control = torch.zeros(self.NUM_EVCS, device=self.device)
        self.voltage_error = torch.zeros(self.NUM_EVCS, device=self.device)
        self.kp_vout = torch.zeros(self.NUM_EVCS, device=self.device)
        self.ki_vout = torch.zeros(self.NUM_EVCS, device=self.device)
        
        # Reset FDI variables
        self.fdi_v = torch.zeros(self.NUM_EVCS, device=self.device)
        self.fdi_i_d = torch.zeros(self.NUM_EVCS, device=self.device)

        # Reset attack-related variables
        self.target_evcs = torch.zeros(self.NUM_EVCS, device=self.device, dtype=torch.int64)
        self.attack_active = False
        self.attack_start_time = torch.tensor(0, device=self.device)
        self.attack_end_time = torch.tensor(0, device=self.device)
        self.attack_duration = torch.tensor(0, device=self.device)
        self.voltage_deviations = torch.zeros(self.NUM_EVCS, device=self.device)
        
        # Initialize state with zeros (keep as numpy for gym compatibility)
        self.state = torch.zeros(25, device=self.device)  # Assuming 25 is the observation space size

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        try:
            # Reset time counter and state
            self.reset_state()
            
            # Get initial state from PINN model
            initial_time = torch.tensor([[0.0]], device=self.device)
            with torch.no_grad():  # Prevent gradient computation
                initial_prediction = self.pinn_model(initial_time)
                if initial_prediction is None:
                    print("Error: Initial prediction is None")
                    return torch.zeros(self.observation_space.shape[0], device=self.device).detach().cpu().numpy(), {}
                    
                evcs_vars = initial_prediction[:, 2 * self.NUM_BUSES:].detach().cpu().numpy()[0]
                initial_state = self.get_observation(evcs_vars)
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            return initial_state.detach().cpu().numpy(), {}  # Return state and empty info dict
            
        except Exception as e:
            print(f"Error in environment reset: {str(e)}")
            return torch.zeros(self.observation_space.shape[0], device=self.device).detach().cpu().numpy(), {}

    def apply_wac_control(self):
        """Apply Wide Area Control with anti-windup protection."""
        self.wac_control = torch.zeros(self.NUM_EVCS, device=self.device)
        for i in range(self.NUM_EVCS):
            # Update integral term with anti-windup
            self.wac_integral[i] = torch.clamp(
                self.wac_integral[i] + self.voltage_error[i] * self.TIME_STEP,
                -self.control_saturation, 
                self.control_saturation
            )
            
            # Calculate control action
            self.wac_control[i] = (
                self.kp_vout[i] * self.voltage_error[i] +
                self.ki_vout[i] * self.wac_integral[i]
            )
            
            # Clip control action
            self.wac_control[i] = torch.clamp(
                self.wac_control[i],
                0,
                1
            )

    def validate_physics(self, new_state):
        """Validate physics constraints."""
        try:
            # Handle scalar input
            if torch.is_tensor(new_state) and new_state.ndim == 0:
                return True
                
            # Convert to tensor if needed
            if not torch.is_tensor(new_state):
                new_state = torch.tensor(new_state, device=self.device)
                
            v_out = new_state[:self.NUM_EVCS]
            i_out = new_state[3*self.NUM_EVCS:4*self.NUM_EVCS]
            i_dc = new_state[4*self.NUM_EVCS:5*self.NUM_EVCS]
            
            voltage_valid = torch.all((v_out >= self.voltage_limits[0]) & 
                                    (v_out <= self.voltage_limits[1]))
            current_valid = torch.all((i_out >= self.current_limits[0]) & 
                                    (i_out <= self.current_limits[1]) &
                                    (i_dc >= self.current_limits[0]) & 
                                    (i_dc <= self.current_limits[1]))
            
            return voltage_valid.item() and current_valid.item()
            
        except Exception as e:
            print(f"Error in validate_physics: {e}")
            return False

    def calculate_rewards(self, voltage_deviations):
        """Calculate rewards for all agents."""
        try:
            # Convert input to tensor if needed
            if not torch.is_tensor(voltage_deviations):
                voltage_deviations = torch.tensor(voltage_deviations, device=self.device)
                
            attack_reward = []
            defender_reward = []
            self.voltage_deviations = voltage_deviations

            for i, deviation in enumerate(self.voltage_deviations):
                if self.target_evcs[i] == 1:  # Only consider targeted EVCSs
                    if deviation > 0.1:
                        attack_reward.append(100 - 0.1 * self.current_time)
                        defender_reward.append(-0.1 * self.current_time - deviation)
                    else:
                        attack_reward.append(0.1 * self.current_time + deviation)
                        defender_reward.append(-0.01 * self.current_time)
                else:
                    attack_reward.append(0)
                    defender_reward.append(0)

            # Convert lists to tensors for calculations
            attack_reward = torch.tensor(attack_reward, device=self.device)
            defender_reward = torch.tensor(defender_reward, device=self.device)
            total_reward = torch.sum(attack_reward + defender_reward)
            
            return total_reward.item()
            
        except Exception as e:
            print(f"Error in calculate_rewards: {e}")
            return 0.0


    def calculate_rewards(self, voltage_deviations):
        """Calculate rewards for all agents."""
        try:
            # Convert input to tensor if needed
            if not torch.is_tensor(voltage_deviations):
                voltage_deviations = torch.tensor(voltage_deviations, device=self.device)
                
            attack_rewards = []
            defender_rewards = []
            
            # Calculate rewards for each EVCS
            for i, deviation in enumerate(voltage_deviations):
                if self.target_evcs[i] == 1:  # For targeted EVCSs
                    if deviation > 0.1:  # Significant deviation
                        # Attacker gets positive reward for successful attack
                        attack_reward = 10.0 * deviation - 0.1 * self.current_time
                        # Defender gets negative reward proportional to deviation
                        defend_reward = -20.0 * deviation - 0.01 * self.current_time
                    else:  # Small deviation
                        # Attacker gets small reward for attempting
                        attack_reward = 0.1 * deviation
                        # Defender gets positive reward for maintaining stability
                        defend_reward = 5.0 * (0.05 - deviation) + 0.01 * self.current_time
                else:  # For non-targeted EVCSs
                    attack_reward = 0.0
                    # Defender gets small positive reward for maintaining stability
                    defend_reward = 1.0 * (0.05 - deviation) if deviation < 0.05 else -5.0 * deviation
                    
                attack_rewards.append(attack_reward)
                defender_rewards.append(defend_reward)
            
            # Convert lists to tensors and sum
            attack_rewards = torch.tensor(attack_rewards, device=self.device)
            defender_rewards = torch.tensor(defender_rewards, device=self.device)
            total_attack_reward = float(torch.sum(attack_rewards))
            total_defend_reward = float(torch.sum(defender_rewards))
            
            # Add time-based penalties
            if self.attack_active:
                time_penalty = 0.05 * self.current_time
                total_attack_reward -= time_penalty
                total_defend_reward += 0.02 * self.current_time
            
            print(f"Attack reward: {total_attack_reward} and Defender reward: {total_defend_reward}")
            
            return {
                'attacker': total_attack_reward,
                'defender': total_defend_reward
            }
            
        except Exception as e:
            print(f"Error in calculate_rewards: {e}")
            return {'attacker': 0.0, 'defender': 0.0}

    def prepare_defender_actions_for_pinn(self, defender_action):
        """Prepare defender actions in the format expected by PINN model."""
        try:
            # Convert input to tensor if needed
            if not torch.is_tensor(defender_action):
                defender_action = torch.tensor(defender_action, device=self.device)
                
            # Split defender action into Kp and Ki adjustments
            kp_adjustments = defender_action[:self.NUM_EVCS]
            ki_adjustments = defender_action[self.NUM_EVCS:]

            # Scale the defender actions to appropriate ranges
            kp_vout = torch.clamp(kp_adjustments, 0.0, self.CONTROL_MAX)
            ki_vout = torch.clamp(ki_adjustments, 0.0, self.CONTROL_MAX)

            # Combine and reshape for PINN model
            wac_params = torch.cat([kp_vout, ki_vout]).unsqueeze(0)
            
            return wac_params
            
        except Exception as e:
            print(f"Error in prepare_defender_actions_for_pinn: {e}")
            return torch.zeros((1, self.NUM_EVCS * 2), device=self.device)

    def step(self, action):
        """Execute one time step within the environment."""
        try:
            with torch.no_grad():  # Prevent gradient computation
                # Update time
                self.time_step_counter += 1
                self.current_time += self.TIME_STEP

                # Validate action dictionary
                if not isinstance(action, dict):
                    raise ValueError(f"Expected dict action, got {type(action)}")
                if not all(k in action for k in ['dqn', 'attacker', 'defender']):
                    raise ValueError("Action dict missing required keys")

                # Convert actions to tensors
                dqn_action = self.decode_action(torch.tensor(action['dqn'], device=self.device))
                attacker_action = torch.tensor(action['attacker'], dtype=torch.float32, device=self.device)
                defender_action = torch.tensor(action['defender'], dtype=torch.float32, device=self.device)

                # Process DQN action
                self.target_evcs = dqn_action[:-1].to(torch.int64)
                self.attack_duration = int(dqn_action[-1]) * 10

                # Set attack parameters
                if torch.any(self.target_evcs > 0):
                    self.attack_active = True
                    if self.attack_start_time == 0:
                        self.attack_start_time = self.time_step_counter
                        self.attack_end_time = torch.clamp(
                            torch.tensor(self.attack_start_time + self.attack_duration, device=self.device),
                            0, 1000
                        )
                else:
                    self.attack_active = False
                    self.attack_start_time = torch.tensor(0, device=self.device)
                    self.attack_end_time = torch.tensor(0, device=self.device)

                # Get PINN prediction
                current_time = torch.tensor([[self.current_time]], device=self.device)
                prediction = self.pinn_model(current_time)
                evcs_vars = prediction[:, 2 * self.NUM_BUSES:].detach().cpu().numpy().flatten()
                current_state = self.get_observation(evcs_vars)

                # Apply actions
                if (self.attack_active and 
                    self.attack_start_time <= self.time_step_counter <= self.attack_end_time):
                    for t in range(self.attack_start_time, self.attack_end_time):
                        current_state = self.apply_attack_effects(
                            current_state, attacker_action.detach(), self.target_evcs, self.attack_duration
                        )
                        current_state = self.apply_defender_actions(current_state, defender_action.detach())

            
            # Update state
            self.state = current_state

            self.voltage_deviations = np.abs(self.state[:self.NUM_EVCS] - 1.0)
            max_deviations= np.max(self.voltage_deviations)

            self.rewards = self.calculate_rewards(self.voltage_deviations)

            
            # Check if episode is done
            done = self.time_step_counter >= 1000 or max_deviations>= 0.5
            
            # Get info
            info = self.get_info(self.voltage_deviations, self.target_evcs, self.attack_duration, self.rewards)
            
            return self.state, self.rewards, done, False, info

        except Exception as e:
            print(f"Error in step: {str(e)}")
            return (
                self.state.detach().cpu().numpy(),
                0.0, 
                True, 
                False, 
                {}
            )

    def apply_attack_effects(self, state, attack_action, target_evcs, attack_duration):
        """Apply attack effects with proper shape handling."""
        try:
            # Convert inputs to tensors if needed
            if not torch.is_tensor(state):
                state = torch.tensor(state, device=self.device)
            if not torch.is_tensor(attack_action):
                attack_action = torch.tensor(attack_action, device=self.device).flatten()
            if not torch.is_tensor(target_evcs):
                target_evcs = torch.tensor(target_evcs, dtype=torch.bool, device=self.device)
            
            # Ensure attack_action has correct shape
            if attack_action.shape[0] != self.NUM_EVCS * 2:
                print(f"Reshaping attack_action from {attack_action.shape} to ({self.NUM_EVCS * 2},)")
                attack_action = torch.resize_(attack_action, (self.NUM_EVCS * 2,))
            
            # Split attack actions
            voltage_attacks = attack_action[:self.NUM_EVCS]
            current_attacks = attack_action[self.NUM_EVCS:]
            
            # Apply attacks only to targeted EVCSs
            state_copy = state.clone()
            for i in range(self.NUM_EVCS):
                if target_evcs[i]:
                    # Apply voltage attack
                    state_copy[i] = torch.clamp(
                        state[i] * (1 + voltage_attacks[i]),
                        self.voltage_limits[0],
                        self.voltage_limits[1]
                    )
                    
                    # Apply current attack
                    current_idx = 3 * self.NUM_EVCS + i
                    state_copy[current_idx] = torch.clamp(
                        state[current_idx] * (1 + current_attacks[i]),
                        self.current_limits[0],
                        self.current_limits[1]
                    )
            
            return state_copy
            
        except Exception as e:
            print(f"Error in apply_attack_effects: {e}")
            return state

    def apply_defender_actions(self, state, defender_action):
        """Apply defender actions (WAC parameter adjustments)."""
        try:
            # Convert inputs to tensors
            if not torch.is_tensor(state):
                state = torch.tensor(state, device=self.device)
            if not torch.is_tensor(defender_action):
                defender_action = torch.tensor(defender_action, device=self.device).flatten()
            
            # Ensure defender_action has correct shape
            if defender_action.shape[0] != self.NUM_EVCS * 2:
                print(f"Reshaping defender_action from {defender_action.shape} to ({self.NUM_EVCS * 2},)")
                defender_action = torch.resize_(defender_action, (self.NUM_EVCS * 2,))
            
            # Split defender action
            kp_adjustments = defender_action[:self.NUM_EVCS]
            ki_adjustments = defender_action[self.NUM_EVCS:]

            # Update WAC parameters
            self.kp_vout = torch.clamp(
                self.WAC_KP_VOUT_DEFAULT + kp_adjustments,
                0.0,
                self.control_saturation
            )
            self.ki_vout = torch.clamp(
                self.WAC_KI_VOUT_DEFAULT + ki_adjustments,
                0.0,
                self.control_saturation
            )

            # Calculate voltage error
            v_out = state[:self.NUM_EVCS]
            self.voltage_error = self.WAC_VOUT_SETPOINT - v_out

            # Apply WAC control
            self.apply_wac_control()

            # Apply control effect
            state_copy = state.clone()
            state_copy[:self.NUM_EVCS] = torch.clamp(
                v_out * (1 + self.wac_control),
                self.voltage_limits[0],
                self.voltage_limits[1]
            )

            return state_copy

        except Exception as e:
            print(f"Error in apply_defender_actions: {e}")
            return state

    def get_info(self, voltage_deviations, target_evcs, attack_duration, rewards):
        """Get current environment info."""
        try:
            # Calculate deviations
            self.cumulative_deviation = torch.sum(voltage_deviations)
            self.voltage_deviations = voltage_deviations
            
            return {
                'time_step': self.time_step_counter,
                'current_time': float(self.current_time),
                'attack_active': self.attack_active,
                'cumulative_deviation': float(self.cumulative_deviation),
                'target_evcs': target_evcs.cpu().numpy().astype(int),
                'attack_duration': int(attack_duration),
                'voltage_deviations': self.voltage_deviations.cpu().numpy(),
                'rewards': rewards
            }
            
        except Exception as e:
            print(f"Error in get_info: {e}")
            return {
                'time_step': self.time_step_counter,
                'current_time': float(self.current_time),
                'attack_active': False,
                'cumulative_deviation': 0.0,
                'target_evcs': [0] * self.NUM_EVCS,
                'attack_duration': 0,
                'voltage_deviations': [0.0] * self.NUM_EVCS,
                'rewards': 0.0
            }

    def get_observation(self, evcs_vars):
        """Get observation from EVCS variables."""
        try:
            if not torch.is_tensor(evcs_vars):
                evcs_vars = torch.tensor(evcs_vars, device=self.device)
                
            v_out_values = []
            soc_values = []
            v_dc_values = []
            i_out_values = []
            i_dc_values = []
            
            for i in range(self.NUM_EVCS):
                v_dc = torch.exp(evcs_vars[i * 18 + 2])  # DC link voltage
                v_out = torch.exp(evcs_vars[i * 18 + 4])  # Output voltage
                soc = evcs_vars[i * 18 + 9]  # State of Charge
                i_out = evcs_vars[i * 18 + 16]  # Output current
                i_dc = evcs_vars[i * 18 + 17]  # DC current

                v_dc_values.append(v_dc)
                v_out_values.append(v_out)
                soc_values.append(soc)
                i_out_values.append(i_out)
                i_dc_values.append(i_dc)

            return torch.cat([
                torch.stack(v_out_values),
                torch.stack(soc_values),
                torch.stack(v_dc_values),
                torch.stack(i_out_values),
                torch.stack(i_dc_values)
            ])
            
        except Exception as e:
            print(f"Error in get_observation: {e}")
            return torch.zeros(self.NUM_EVCS * 5, device=self.device)

    def control_saturation(self, value, v_min, v_max):
        """Saturate control signal between minimum and maximum values."""
        try:
            if not torch.is_tensor(value):
                value = torch.tensor(value, device=self.device)
            return torch.clamp(value, v_min, v_max)
        except Exception as e:
            print(f"Error in control_saturation: {e}")
            return torch.tensor(0.0, device=self.device)

    def update_wac_parameters(self, defender_action):
        """Update WAC parameters from defender actions."""
        try:
            if not torch.is_tensor(defender_action):
                defender_action = torch.tensor(defender_action, device=self.device)
                
            # Split defender action
            kp_adjustments = defender_action[:self.NUM_EVCS]
            ki_adjustments = defender_action[self.NUM_EVCS:]

            # Update WAC parameters
            self.kp_vout = self.control_saturation(
                kp_adjustments,
                0.0,
                self.CONTROL_MAX
            )
            self.ki_vout = self.control_saturation(
                ki_adjustments,
                0.0,
                self.CONTROL_MAX
            )
            
        except Exception as e:
            print(f"Error in update_wac_parameters: {e}")

    def decode_action(self, action):
        """
        Decode the DQN action into target EVCSs and attack duration.
        Handles scalar, vector, and 0-dim tensor inputs.
        """
        try:
            # First convert action to a workable format
            if torch.is_tensor(action):
                if action.dim() == 0:  # Handle 0-dim tensor
                    action = int(action.item())
            
            # Now handle different input types
            if isinstance(action, (int, float, np.integer, np.floating)):
                # Convert scalar to our action format
                action_int = int(action)
                
                # Create target vector (NUM_EVCS elements)
                target_evcs = torch.zeros(self.NUM_EVCS, device=self.device)
                
                # Set target EVCS (assuming action_int encodes which EVCS to target)
                evcs_idx = action_int % self.NUM_EVCS
                target_evcs[evcs_idx] = 1.0
                
                # Set duration (0-9 range, scaled by 10)
                duration = torch.tensor([10.0], device=self.device)  # Fixed duration for now
                
            elif isinstance(action, (list, np.ndarray)) or (torch.is_tensor(action) and action.dim() > 0):
                # Convert to tensor if not already
                if not torch.is_tensor(action):
                    action = torch.tensor(action, device=self.device)
                
                # Split into target and duration
                target_evcs = action[:-1].float()
                duration = action[-1:].float() * 10.0  # Scale duration
                
                # Ensure binary targets
                target_evcs = (target_evcs > 0).float()
                
            else:
                raise ValueError(f"Unsupported action type: {type(action)}")

            # Combine and validate output
            decoded_action = torch.cat([target_evcs, duration])
            assert decoded_action.dim() == 1, f"Expected 1-dim tensor, got {decoded_action.dim()}-dim"
            assert len(decoded_action) == self.NUM_EVCS + 1, f"Expected length {self.NUM_EVCS + 1}, got {len(decoded_action)}"
            
            return decoded_action

        except Exception as e:
            print(f"Error in decode_action: {e}")
            print(f"Action type: {type(action)}")
            print(f"Action value: {action}")
            # Return safe default values: no targets and zero duration
            return torch.zeros(self.NUM_EVCS + 1, device=self.device)
