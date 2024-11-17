#Testing with WAC
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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


# Add these constants near the top with other system parameters
MAX_VOLTAGE_PU = 1.2  # Maximum allowable voltage in per unit
MIN_VOLTAGE_PU = 0.8  # Minimum allowable voltage in per unit
VOLTAGE_VIOLATION_PENALTY = 1000.0  # Large penalty for voltage violations


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
EVCS_PLL_KI = 0.5
MAX_PLL_ERROR = 10.0

EVCS_OUTER_KP = 20 #0.5 and 0.3 was original value
EVCS_OUTER_KI = 0.05

EVCS_INNER_KP = 10
EVCS_INNER_KI = 0.06
OMEGA_N = 2 * np.pi * 60  # Nominal angular frequency (60 Hz)

# Wide Area Controller Parameters
WAC_KP_VDC = 10
WAC_KI_VDC = 0.06

WAC_KP_VOUT = 20
WAC_KI_VOUT =0.03 # original value is 0.5
WAC_VOLTAGE_SETPOINT = 1.0 # V_BASE_DC / V_BASE_LV  # Desired DC voltage in p.u.
WAC_VOUT_SETPOINT = 1.0 # V_BASE_DC / V_BASE_LV  # Desired output voltage in p.u.


# Other Parameters
CONSTRAINT_WEIGHT = 1.0
LCL_L1 = 55e-6 / Z_BASE_LV  # Convert to p.u.
LCL_L2 = 30e-6 / Z_BASE_LV  # Convert to p.u.
LCL_CF = 10e-6 * S_BASE / (V_BASE_LV**2)  # Convert to p.u.
R = 0.1 / Z_BASE_LV  # Convert to p.u.
C_dc = 0.01 * S_BASE / (V_BASE_LV**2)  # Convert to p.u.


L_dc = 10e-6 / Z_BASE_LV  # Convert to p.u.
v_battery = 800 / V_BASE_DC  # Convert to p.u.
R_battery = 0.1 / Z_BASE_LV  # Convert to p.u.

# Time parameters
TIME_STEP = 0.1  # 1 ms
TOTAL_TIME = 500  # 100 seconds


POWER_BALANCE_WEIGHT = 1.0
RATE_OF_CHANGE_LIMIT = 0.05  # Maximum 5% change per time step
VOLTAGE_STABILITY_WEIGHT = 2.0
POWER_FLOW_WEIGHT = 1.5
MIN_VOLTAGE_LIMIT = 0.85  # Minimum allowable voltage
THERMAL_LIMIT_WEIGHT = 1.0
COORDINATION_WEIGHT = 0.8  # Weight for coordinated attack impact

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
    [6, 60, 20, 0], [7, 145, 100, 0], [8, 160, 100, 0], [9, 60, 20, 0], [10, 100, 60, 0],
    [11, 100, 30, 0], [12, 60, 35, 0], [13, 60, 35, 0], [14, 80, 80, 0], [15, 100, 60, 0],
    [16, 100, 20, 0], [17, 60, 20, 0], [18, 90, 40, 0], [19, 90, 40, 0], [20, 90, 40, 0],
    [21, 90, 40, 0], [22, 90, 40, 0], [23, 90, 40, 0], [24, 420, 200, 0], [25, 380, 200, 0],
    [26, 100, 25, 0], [27, 60, 25, 0], [28, 60, 20, 0], [29, 120, 70, 0], [30, 160, 510, 0],
    [31, 150, 70, 0], [32, 210, 100, 0], [33, 60, 40, 0]
])

bus_data[:, 1:3] = bus_data[:, 1:3]*10**3 / S_BASE
EVCS_CAPACITY = 80e3 / S_BASE    # Convert kW to per-unit

# Initialize Y-bus matrix
Y_bus = np.zeros((NUM_BUSES, NUM_BUSES), dtype=complex)

# Fill Y-bus matrix
for line in line_data:
    from_bus, to_bus, r, x = line
    from_bus, to_bus = int(from_bus)-1 , int(to_bus)-1 # Convert to 0-based index
    y = 1 / complex(r, x)
    Y_bus[from_bus, from_bus] += y
    Y_bus[to_bus, to_bus] += y
    Y_bus[from_bus, to_bus] -= y
    Y_bus[to_bus, from_bus] -= y

# Convert to TensorFlow constant
Y_bus_tf = tf.constant(Y_bus, dtype=tf.complex64)


attack_vector = tf.Variable(
    tf.zeros([NUM_EVCS], dtype=tf.float32), trainable=True, name="attack_vector"
)



class EVCS_PowerSystem_PINN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.all_weights = []

        # Initial layer
        self.initial_kernel = self.add_weight(name="initial_kernel", shape=(1, 500), initializer='glorot_normal')
        self.initial_bias = self.add_weight(name="initial_bias", shape=(500,), initializer='zeros')
        self.all_weights.extend([self.initial_kernel, self.initial_bias])

        # Dense layers
        self.dense_kernels = []
        self.dense_biases = []
        for i in range(6):
            kernel = self.add_weight(name=f"dense_kernel_{i}", shape=(500, 500), initializer='glorot_normal')
            bias = self.add_weight(name=f"dense_bias_{i}", shape=(500,), initializer='zeros')
            self.dense_kernels.append(kernel)
            self.dense_biases.append(bias)
            self.all_weights.extend([kernel, bias])

        # Output layer
        self.output_kernel = self.add_weight(name="output_kernel", shape=(500, NUM_BUSES * 2 + NUM_EVCS * 18), initializer='glorot_normal')
        self.output_bias = self.add_weight(name="output_bias", shape=(NUM_BUSES * 2 + NUM_EVCS * 18,), initializer='zeros')
        self.all_weights.extend([self.output_kernel, self.output_bias])

    def call(self, t):
        x = tf.matmul(t, self.initial_kernel) + self.initial_bias
        x = tf.nn.tanh(x)

        for kernel, bias in zip(self.dense_kernels, self.dense_biases):
            x = tf.matmul(x, kernel) + bias
            x = tf.nn.tanh(x)

        return tf.matmul(x, self.output_kernel) + self.output_bias

    @property
    def trainable_variables(self):
        return self.all_weights


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

# Add these new constants near the other system parameters
ATTACK_BOUNDS = tf.constant(0.3, dtype=tf.float32)  # Maximum allowed deviation (±30%)
ATTACK_WEIGHT = 2.0  # Weight for attack objective
STABILITY_WEIGHT = 0.5  # Weight for stability constraint
MIN_VOLTAGE_LIMIT = 0.80
LEARNING_RATE_ATTACK = 1e-2  # Separate learning rate for attack optimization
VOLTAGE_CONSTRAINT_WEIGHT = 1.0  # Weight for voltage constraints

# Add these constants near the top with other system parameters
MAX_VOLTAGE_PU = 1.1  # Maximum allowable voltage in per unit
MIN_VOLTAGE_PU = 0.9  # Minimum allowable voltage in per unit
VOLTAGE_VIOLATION_PENALTY = 1000.0  # Large penalty for voltage violations

def physics_loss_with_attack(model, t, Y_bus_tf, bus_data, attack_vector):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        predictions = model(t)

        # Extract variables
        V = safe_op(tf.exp(predictions[:, :NUM_BUSES]))
        theta = safe_op(tf.math.atan(predictions[:, NUM_BUSES:2*NUM_BUSES]))
        evcs_vars = predictions[:, 2*NUM_BUSES:]

        # Power flow equations
        V_complex = tf.complex(V * tf.cos(theta), V * tf.sin(theta))
        S = V_complex[:, :, tf.newaxis] * tf.math.conj(tf.matmul(Y_bus_tf, V_complex[:, :, tf.newaxis]))
        P_calc = tf.math.real(S)[:, :, 0]
        Q_calc = tf.math.imag(S)[:, :, 0]

        P_mismatch = P_calc - bus_data[:, 1]
        Q_mismatch = Q_calc - bus_data[:, 2]

        # Voltage regulation for all buses
        V_nominal = 1.0  # Nominal voltage in p.u.
        V_lower = 0.95  # Lower voltage limit
        V_upper = 1.05  # Upper voltage limit
        V_regulation_loss = safe_op(tf.reduce_mean(tf.square(tf.maximum(0.0, V_lower - V) + tf.maximum(0.0, V - V_upper))))

        evcs_loss = []
        attack_loss = 0.0
        stability_loss = 0.0
        voltage_deviation_loss = 0.0
        power_balance_loss = 0.0
        thermal_limit_loss = 0.0
        coordination_loss = 0.0
        constraint_loss = 0.0
        voltage_violation_loss = 0.0
        voltage_regulation_loss = 0.0
        
        # Store attacked voltages for coordination
        attacked_voltages = []

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


            attack = attack_vector[i]
            # v_out = tf.exp(evcs[:, 4])  # Output voltage
            v_out_attacked = v_out + attack

            v_out_attacked = safe_op(tf.exp(v_out_attacked))
            attacked_voltages.append(v_out_attacked)
            
            # Penalize voltages outside acceptable range
            upper_violation = tf.maximum(0.0, v_out_attacked - MAX_VOLTAGE_PU)
            lower_violation = tf.maximum(0.0, MIN_VOLTAGE_PU - v_out_attacked)

            #removed for benign response plotting

            voltage_violation_loss += tf.reduce_mean(
                VOLTAGE_VIOLATION_PENALTY * (tf.square(upper_violation) + tf.square(lower_violation))
            )

            v_out_regulation_loss = safe_op(tf.reduce_mean(
                tf.square(tf.maximum(0.0, 0.95 - v_out_attacked) + 
                tf.maximum(0.0, v_out - 1.05))
                ))      

            VOLTAGE_REG_WEIGHT = 10.0 

            voltage_regulation_loss += VOLTAGE_REG_WEIGHT * v_out_regulation_loss

            # Clarke and Park Transformations
            v_alpha = v_ac
            v_beta = tf.zeros_like(v_ac)
            i_alpha = i_ac
            i_beta = tf.zeros_like(i_ac)
            v_d = safe_op(v_alpha * tf.cos(delta) + v_beta * tf.sin(delta))
            v_q = safe_op(-v_alpha * tf.sin(delta) + v_beta * tf.cos(delta))
            i_d_measured = safe_op(i_alpha * tf.cos(delta) + i_beta * tf.sin(delta))
            i_q_measured = safe_op(-i_alpha * tf.sin(delta) + i_beta * tf.cos(delta))

            # PLL Dynamics
            v_q_normalized = tf.nn.tanh(safe_op(v_q))
            pll_error = safe_op(EVCS_PLL_KP * v_q_normalized + EVCS_PLL_KI * phi_q)
            pll_error = tf.clip_by_value(pll_error, -MAX_PLL_ERROR, MAX_PLL_ERROR)

            # Wide Area Controller
            v_dc_actual = v_dc *   (V_BASE_DC/V_BASE_LV)
            v_out_actual = v_out * (V_BASE_DC/V_BASE_LV)
            # Modified attack impact calculation

            
            # Calculate voltage deviation from nominal
            voltage_deviation = tf.abs(v_out_attacked - WAC_VOUT_SETPOINT)
            impact_scale = tf.exp(voltage_deviation * 2.0) 

            diversity_term = tf.reduce_mean(tf.square(
            attack_vector - tf.roll(attack_vector, shift=1, axis=0)
            ))

            wac_error_vdc += WAC_VOLTAGE_SETPOINT - v_dc

            wac_integral_vdc += wac_error_vdc * TIME_STEP
            wac_output_vdc = WAC_KP_VDC * wac_error_vdc + WAC_KI_VDC * wac_integral_vdc

            v_dc_ref = WAC_VOLTAGE_SETPOINT + wac_output_vdc

            wac_error_vout += WAC_VOUT_SETPOINT - v_out_attacked
            wac_integral_vout += wac_error_vout * TIME_STEP
            wac_output_vout = WAC_KP_VOUT * wac_error_vout + WAC_KI_VOUT * wac_integral_vout

            v_out_ref = WAC_VOUT_SETPOINT + wac_output_vout
            v_out_loss = tf.reduce_mean(tf.square(v_out_attacked - v_out_ref))

            attack_loss += -ATTACK_WEIGHT * tf.reduce_mean(
            voltage_deviation * impact_scale
            ) + 0.1 * diversity_term

            min_attack_magnitude = 0.05  # Minimum desired attack magnitude
            attack_magnitude_loss = tf.reduce_mean(
            tf.maximum(0.0, min_attack_magnitude - tf.abs(attack))
            )
            attack_loss += 0.5 * attack_magnitude_loss

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

            duty_cycle = tf.clip_by_value(v_out_ref / v_dc, 0, 1)
            v_out_expected = duty_cycle * v_dc

            v_out_actual = v_out
            v_out_lower = V_OUT_NOMINAL * (1 - V_OUT_VARIATION)
            v_out_upper = V_OUT_NOMINAL * (1 + V_OUT_VARIATION)
            v_out_constraint = safe_op(tf.reduce_mean(tf.square(tf.maximum(0.0, v_out_lower - v_out_actual) + tf.maximum(0.0, v_out_actual - v_out_upper))))

            v_out_nominal = V_OUT_NOMINAL
          
            lower_bound = v_out_nominal * 0.80
            upper_bound = v_out_nominal * 1.2
            constraint_loss += tf.reduce_mean(tf.square(tf.maximum(0.0, v_out_attacked - upper_bound) + tf.maximum(0.0, lower_bound - v_out_attacked)))

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

            # Calculate voltage deviation from nominal
            voltage_deviation = tf.abs(v_out_attacked - WAC_VOUT_SETPOINT)
            
            # Penalize rapid changes in voltage (stability term)
            with tf.GradientTape() as temp_tape:
                temp_tape.watch(t)
                dv_out_dt = temp_tape.gradient(v_out_attacked, t)
                if dv_out_dt is not None:
                    stability_loss += tf.reduce_mean(
                        tf.square(tf.maximum(0.0, tf.abs(dv_out_dt) - tf.constant(RATE_OF_CHANGE_LIMIT, dtype=tf.float32))))

            # 2. Power Balance Constraint
            P_dc = safe_op(v_dc * i_dc)
            P_out = safe_op(v_out_attacked * i_out)
            power_balance_loss += tf.reduce_mean(tf.square(P_dc - P_out))

            # 3. Thermal Limit Constraint (current-based)
            i_limit = 1.2  # 120% of nominal current
            thermal_limit_loss += safe_op(tf.reduce_mean(
                tf.square(tf.maximum(0.0, tf.abs(i_out) - i_limit))))

            # 4. Minimum Voltage Constraint
            voltage_deviation_loss += safe_op(tf.reduce_mean(
                tf.square(tf.maximum(0.0, MIN_VOLTAGE_PU - v_out_attacked))))

            # 5. Modified Attack Impact
            nominal_voltage = WAC_VOUT_SETPOINT
            voltage_deviation = tf.abs(v_out_attacked - nominal_voltage)
            attack_loss += -ATTACK_WEIGHT * tf.reduce_mean(
                voltage_deviation * tf.exp(-STABILITY_WEIGHT * tf.square(voltage_deviation)))

            # Modified attack objective (maximize deviation while maintaining stability)
            attack_loss += -ATTACK_WEIGHT * tf.reduce_mean(voltage_deviation) + STABILITY_WEIGHT * stability_loss

            # Voltage constraint (keep within bounds)
            v_out_constraint = tf.reduce_mean(
            tf.square(tf.maximum(0.0, v_out_attacked - WAC_VOUT_SETPOINT * (1 + ATTACK_BOUNDS))) + \
            tf.square(tf.maximum(0.0, WAC_VOUT_SETPOINT * (1 - ATTACK_BOUNDS) - v_out_attacked))
            )
            voltage_deviation_loss += VOLTAGE_CONSTRAINT_WEIGHT * v_out_constraint

            # Append all losses
            evcs_loss.extend([
                ddelta_dt_loss, domega_dt_loss, dphi_d_dt_loss, dphi_q_dt_loss,
                di_d_dt_loss, di_q_dt_loss, di_L1_dt_loss, di_L2_dt_loss, dv_c_dt_loss,
                dv_dc_dt_loss, v_out_loss, v_out_constraint, di_out_dt_loss, dsoc_dt_loss,
                power_balance_loss, current_consistency_loss
            ])

        # Combine losses
        power_flow_loss = safe_op(tf.reduce_mean(tf.square(P_mismatch) + tf.square(Q_mismatch)))
        evcs_total_loss = safe_op(tf.reduce_sum(evcs_loss))
        wac_loss = safe_op(tf.reduce_mean(tf.square(wac_error_vdc) + tf.square(wac_error_vout)))
        V_regulation_loss = safe_op(tf.reduce_mean(V_regulation_loss))

        attacked_voltages = tf.stack(attacked_voltages)
        mean_attack = tf.reduce_mean(attacked_voltages)
        coordination_loss += tf.reduce_mean(
        tf.square(attacked_voltages - mean_attack)
        )
        
        # Modified total loss calculation
        total_loss = (
            power_flow_loss * POWER_FLOW_WEIGHT +
            evcs_total_loss +
            CONSTRAINT_WEIGHT * wac_loss +
            V_regulation_loss +
            VOLTAGE_REG_WEIGHT * v_out_regulation_loss +  # Added weighted voltage regulation
            attack_loss +
            STABILITY_WEIGHT * stability_loss +
            POWER_BALANCE_WEIGHT * power_balance_loss +
            THERMAL_LIMIT_WEIGHT * thermal_limit_loss +
            voltage_deviation_loss +
            COORDINATION_WEIGHT * coordination_loss +
            voltage_violation_loss
        )

    del tape

    return total_loss

class AttackOptimizer:
    def __init__(self, num_evcs):
        self.momentum = tf.Variable(tf.zeros([num_evcs], dtype=tf.float32), trainable=False)
        self.beta = tf.constant(0.9, dtype=tf.float32)
        self.prev_attack = tf.Variable(tf.zeros([num_evcs], dtype=tf.float32), trainable=False)

    @tf.function
    def apply_gradients(self, grad_vars):
        grad, attack_vector = grad_vars[0]
        
        # Ensure gradients have correct shape
        grad = tf.reshape(grad, [NUM_EVCS])
        
        # Update momentum
        self.momentum.assign(self.beta * self.momentum + (1 - self.beta) * grad)
        
        # Calculate new attack
        new_attack = attack_vector - tf.constant(LEARNING_RATE_ATTACK, dtype=tf.float32) * self.momentum
        
        # Apply bounds
        bounded_attack = tf.clip_by_value(
            new_attack,
            clip_value_min=-ATTACK_BOUNDS * WAC_VOUT_SETPOINT,
            clip_value_max=ATTACK_BOUNDS * WAC_VOUT_SETPOINT
        )
        
        return bounded_attack

def evaluate_model(model, attack_vector=None):
    """
    Evaluate the model's performance with or without attack
    """
    # Generate test time points
    t_test = tf.linspace(0.0, TOTAL_TIME, 100)
    t_test = tf.reshape(t_test, [-1, 1])
    
    # Get predictions
    predictions = model(t_test)
    
    # Extract voltages and angles
    V = tf.exp(predictions[:, :NUM_BUSES])
    theta = tf.math.atan(predictions[:, NUM_BUSES:2*NUM_BUSES])
    
    # Extract EVCS variables
    evcs_vars = predictions[:, 2*NUM_BUSES:]
    
    print("\nModel Evaluation Results:")
    print("-------------------------")
    
    # Voltage profile
    V_mean = tf.reduce_mean(V, axis=0)
    V_std = tf.math.reduce_std(V, axis=0)
    print(f"\nVoltage Profile:")
    print(f"Mean voltages across buses: {V_mean.numpy()}")
    print(f"Voltage standard deviation: {V_std.numpy()}")
    
    # EVCS Analysis
    for i, bus in enumerate(EVCS_BUSES):
        evcs = evcs_vars[:, i*18:(i+1)*18]
        v_out = tf.exp(evcs[:, 4])  # Output voltage
        i_out = evcs[:, 5]  # Output current
        
        if attack_vector is not None:
            v_out = v_out + attack_vector[i]
        
        print(f"\nEVCS {i} at Bus {bus}:")
        print(f"Average output voltage: {tf.reduce_mean(v_out).numpy():.4f} p.u.")
        print(f"Average output current: {tf.reduce_mean(i_out).numpy():.4f} p.u.")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Voltage profile
    plt.subplot(2, 1, 1)
    plt.plot(range(NUM_BUSES), V_mean.numpy(), 'b-', label='Mean Voltage')
    plt.fill_between(range(NUM_BUSES), 
                    V_mean.numpy() - V_std.numpy(), 
                    V_mean.numpy() + V_std.numpy(), 
                    alpha=0.2)
    plt.grid(True)
    plt.xlabel('Bus Number')
    plt.ylabel('Voltage (p.u.)')
    plt.title('Voltage Profile')
    plt.legend()
    
    # EVCS output voltages
    plt.subplot(2, 1, 2)
    for i, bus in enumerate(EVCS_BUSES):
        evcs = evcs_vars[:, i*18:(i+1)*18]
        v_out = tf.exp(evcs[:, 4])
        if attack_vector is not None:
            v_out = v_out + attack_vector[i]
        plt.plot(t_test[:, 0], v_out, label=f'EVCS {i}')
    
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Output Voltage (p.u.)')
    plt.title('EVCS Output Voltages')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

class ModelTrainer:
    def __init__(self, num_evcs, initial_learning_rate=1e-6):
        self.model = EVCS_PowerSystem_PINN()
        self.attack_vector = tf.Variable(
            tf.random.uniform([num_evcs], minval=-0.04, maxval=0.04, dtype=tf.float32),
            trainable=True,
            name="attack_vector"
        )

        # #used for benign response plotting  
        # self.attack_vector = tf.Variable(
        #     tf.zeros([num_evcs], dtype=tf.float32),
        #     trainable=False,  # Make it non-trainable
        #     name="attack_vector"
        # )
        
        # Modified optimizers with only clipnorm
        self.model_optimizer = tf.keras.optimizers.SGD(
            learning_rate=initial_learning_rate,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0  # Only use clipnorm
        )
        
        self.attack_optimizer = tf.keras.optimizers.SGD(
            learning_rate=1e-6,
            momentum=0.9,
            clipnorm=1.0  # Only use clipnorm
        )
        
        self.best_attack = tf.Variable(
            tf.zeros([num_evcs], dtype=tf.float32),
            trainable=False
        )
        self.best_attack_impact = tf.Variable(-1.0, dtype=tf.float32)
        
        # Gradient scaling factor
        self.grad_scale = tf.Variable(1.0, dtype=tf.float32)

        # Learning rate scheduler
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.95,
            staircase=True
        )

        # Add these new tracking variables
        self.attack_history = []
        self.voltage_history = []
        
    @tf.function
    def train_step(self, t_batch, Y_bus_tf, bus_data):
        with tf.GradientTape(persistent=True) as tape:
            # Scale predictions for numerical stability
            predictions = self.model(t_batch)
            predictions = tf.clip_by_value(predictions, -10.0, 10.0)
            
            # Track voltages before attack
            evcs_vars = predictions[:, 2*NUM_BUSES:]
            original_voltages = []
            attacked_voltages = []
            
            for i, bus in enumerate(EVCS_BUSES):
                v_out = tf.exp(evcs_vars[:, i*18 + 4])
                v_out_attacked = v_out + self.attack_vector[i]
                original_voltages.append(v_out)
                attacked_voltages.append(v_out_attacked)
            
            # Store the current state (not part of the graph)
            tf.py_function(
                self._update_history,
                [self.attack_vector, tf.stack(original_voltages), tf.stack(attacked_voltages), t_batch],
                []
            )
            
            # Calculate loss with additional stability measures
            physics_loss = physics_loss_with_attack(
                self.model, t_batch, Y_bus_tf, bus_data, self.attack_vector
            )
            
            # Add L2 regularization
            l2_reg = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            total_loss = physics_loss + l2_reg
            
            attack_impact = self.calculate_attack_impact(t_batch)

        # Process gradients with additional safety checks
        def process_gradients(grads):
            if grads is None:
                return None
            # Remove NaN/Inf values
            grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
            # Scale large gradients
            grad_norm = tf.norm(grads)
            grads = tf.where(grad_norm > 1.0, grads / grad_norm, grads)
            return grads

        # Get and process gradients
        model_gradients = tape.gradient(total_loss, self.model.trainable_variables)
        attack_gradients = tape.gradient(total_loss, self.attack_vector)

        # Process gradients
        model_gradients = [process_gradients(g) for g in model_gradients]
        attack_gradients = process_gradients(attack_gradients)

        # Apply gradients only if they are valid
        if all(g is not None for g in model_gradients):
            self.model_optimizer.apply_gradients(
                zip(model_gradients, self.model.trainable_variables)
            )

        if attack_gradients is not None:
            self.attack_optimizer.apply_gradients(
                [(attack_gradients, self.attack_vector)]
            )
            # Clip attack vector
            self.attack_vector.assign(
                tf.clip_by_value(self.attack_vector, -0.1, 0.1)
            )

        # Update best attack if necessary
        update_condition = tf.logical_and(
            attack_impact > self.best_attack_impact,
            tf.math.is_finite(attack_impact)
        )
        
        self.best_attack.assign(
            tf.where(update_condition, self.attack_vector, self.best_attack)
        )
        self.best_attack_impact.assign(
            tf.where(update_condition, attack_impact, self.best_attack_impact)
        )

        return total_loss, attack_impact

    def _update_history(self, attack_vector, original_voltages, attacked_voltages, timestamps):
        """Helper function to store history (called by py_function)"""
        self.attack_history.append({
            'time': timestamps.numpy(),
            'attack_vector': attack_vector.numpy(),
            'original_voltages': original_voltages.numpy(),
            'attacked_voltages': attacked_voltages.numpy()
        })

    def plot_attack_history(self):
        """Plot the attack vector and voltage evolution over time"""
        if not self.attack_history:
            print("No history available to plot")
            return

        # Convert history to numpy arrays and use indices as time steps
        num_steps = len(self.attack_history)
        time_steps = np.arange(num_steps)
        attacks = np.array([h['attack_vector'] for h in self.attack_history])
        orig_v = np.array([np.mean(h['original_voltages'], axis=1) for h in self.attack_history])
        attacked_v = np.array([np.mean(h['attacked_voltages'], axis=1) for h in self.attack_history])

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot attack vectors
        for i in range(NUM_EVCS):
            ax1.plot(time_steps, attacks[:, i], label=f'EVCS {i} at Bus {EVCS_BUSES[i]}')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Attack Magnitude')
        ax1.set_title('Attack Vector Evolution')
        ax1.legend()
        ax1.grid(True)

        # Plot voltage changes
        for i in range(NUM_EVCS):
            ax2.plot(time_steps, orig_v[:, i], '--', label=f'Original V{i}')
            ax2.plot(time_steps, attacked_v[:, i], '-', label=f'Attacked V{i}')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Voltage Deviation (V)')
        ax2.set_title('Voltage Evolution')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Try to save results to file, with error handling
        final_results = {
            'attack_vector': attacks[-1],
            'original_voltages': orig_v[-1],
            'attacked_voltages': attacked_v[-1],
            'EVCS_buses': EVCS_BUSES
        }
        
        try:
            # Try to save in current directory
            np.save('attack_results.npy', final_results)
        except PermissionError:
            try:
                # Try to save in user's home directory
                import os
                home_dir = os.path.expanduser("~")
                save_path = os.path.join(home_dir, 'attack_results.npy')
                np.save(save_path, final_results)
                print(f"\nResults saved to: {save_path}")
            except Exception as e:
                print(f"\nCouldn't save results to file: {str(e)}")
                print("Displaying results in console only.")

        # Print final results
        print("\nFinal Results:")
        print(f"Attack Vector: {attacks[-1]}")
        for i in range(NUM_EVCS):
            print(f"EVCS {i} at Bus {EVCS_BUSES[i]}:")
            print(f"  Original Voltage: {orig_v[-1][i]:.4f}")
            print(f"  Attacked Voltage: {attacked_v[-1][i]:.4f}")
            print(f"  Voltage Change: {(attacked_v[-1][i] - orig_v[-1][i]):.4f}")

    def calculate_attack_impact(self, t_batch):
        """Calculate the impact of current attack vector"""
        predictions = self.model(t_batch)
        evcs_vars = predictions[:, 2*NUM_BUSES:]
        
        total_impact = tf.constant(0.0, dtype=tf.float32)
        for i, bus in enumerate(EVCS_BUSES):
            v_out = tf.exp(evcs_vars[:, i*18 + 4])
            v_out_attacked = v_out + self.attack_vector[i]
            impact = tf.reduce_mean(tf.abs(v_out_attacked - v_out))
            total_impact += impact
            
        return total_impact

    def get_best_attack(self):
        """Return the best attack vector found during training"""
        return self.best_attack.numpy() if self.best_attack is not None else None




def train_model_with_attack(epochs=5000, batch_size=128):  # Reduced epochs and batch size
    # Initialize trainer
    trainer = ModelTrainer(NUM_EVCS)
    
    # Convert bus_data to tensor once
    bus_data_tf = tf.constant(bus_data, dtype=tf.float32)

    # Training loop
    best_attack = None
    best_impact = float('-inf')
    
    # Add early stopping
    patience = 500
    min_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(epochs):
            t_batch = tf.random.uniform((batch_size, 1), minval=0, maxval=TOTAL_TIME)
            loss, impact = trainer.train_step(t_batch, Y_bus_tf, bus_data_tf)

            # Track best attack
            if impact > best_impact:
                best_impact = impact
                best_attack = tf.identity(trainer.attack_vector)

            # Early stopping logic
            if loss < min_loss:
                min_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

            if epoch % 100 == 0:  # Reduced logging frequency
                tf.print(f"Epoch {epoch}, Loss: {loss:.6f}, "
                        f"Impact: {impact:.6f}, "
                        f"Current Attack Vector: {trainer.attack_vector.numpy()}")

    except Exception as e:
        tf.print(f"Error occurred during epoch {epoch}: {str(e)}")
        
    if best_attack is not None:
        tf.print("\nTraining completed!")
        tf.print("Final Best Attack Vector:", best_attack.numpy())
        tf.print("Final Best Attack Impact:", best_impact)
    else:
        tf.print("\nTraining completed but no valid attack vector was found.")

    # Add plotting at the end
    trainer.plot_attack_history()
    
    return trainer.model, best_attack

def evaluate_attack(model, attack_vector, t, Y_bus_tf, bus_data):
    """Evaluate the impact of an attack vector"""
    predictions = model(t)
    evcs_vars = predictions[:, 2*NUM_BUSES:]
    
    print("\nAttack Evaluation:")
    for i, bus in enumerate(EVCS_BUSES):
        v_out = tf.exp(evcs_vars[:, i*18 + 4])
        v_out_attacked = v_out + attack_vector[i]
        impact = tf.reduce_mean(tf.abs(v_out_attacked - v_out))
        print(f"EVCS {i} at Bus {bus}: Impact = {impact:.4f}")

def plot_attack_vector(attack_vector):
    """Plot the attack vector"""
    plt.figure(figsize=(10, 6))
    plt.bar(EVCS_BUSES, attack_vector.numpy())
    plt.xlabel('EVCS Bus Number')
    plt.ylabel('Attack Magnitude')
    plt.title('Attack Vector Distribution')
    plt.grid(True)
    plt.show()

def evaluate_model_with_plots(model, attack_vector=None):
        """
        Evaluate the model's performance and create plots with or without attack
        """
        # Generate test time points
        t_test = tf.linspace(0.0, TOTAL_TIME, 1000)  # Increased points for smoother plots
        t_test = tf.reshape(t_test, [-1, 1])
        
        # Get predictions
        predictions = model(t_test)
        
        # Extract EVCS variables
        evcs_vars = predictions[:, 2*NUM_BUSES:]
        
        # Create figure for voltage outputs
        plt.figure(figsize=(12, 6))
        
        # Plot title will indicate if this is with or without attack
        title_suffix = "with Attack" if attack_vector is not None else "without Attack"
        plt.title(f'EVCS Output Voltages {title_suffix}')
        
        # Plot each EVCS output voltage
        for i, bus in enumerate(EVCS_BUSES):
            evcs = evcs_vars[:, i*18:(i+1)*18]
            v_out = tf.exp(evcs[:, 4])  # Output voltage
            
            if attack_vector is not None:
                v_out = v_out + attack_vector[i]
            
            plt.plot(t_test[:, 0], v_out, label=f'EVCS {i} at Bus {bus}')
        
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Output Voltage (p.u.)')
        plt.legend()
        
        # Add horizontal lines for voltage limits
        plt.axhline(y=MAX_VOLTAGE_PU, color='r', linestyle='--', alpha=0.3, label='Upper Limit')
        plt.axhline(y=MIN_VOLTAGE_PU, color='r', linestyle='--', alpha=0.3, label='Lower Limit')
        
        plt.tight_layout()
        
        # Print analysis results
        print(f"\nVoltage Analysis {title_suffix}:")
        print("-------------------------")
        for i, bus in enumerate(EVCS_BUSES):
            evcs = evcs_vars[:, i*18:(i+1)*18]
            v_out = tf.exp(evcs[:, 4])
            if attack_vector is not None:
                v_out = v_out + attack_vector[i]
            
            v_mean = tf.reduce_mean(v_out)
            v_std = tf.math.reduce_std(v_out)
            v_max = tf.reduce_max(v_out)
            v_min = tf.reduce_min(v_out)
            
            print(f"\nEVCS {i} at Bus {bus}:")
            print(f"  Mean voltage: {v_mean:.4f} p.u.")
            print(f"  Std deviation: {v_std:.4f} p.u.")
            print(f"  Max voltage: {v_max:.4f} p.u.")
            print(f"  Min voltage: {v_min:.4f} p.u.")

# Main execution
if __name__ == '__main__':
    try:
        # Train the model and get the best attack vector
        model, best_attack = train_model_with_attack()

        # Evaluate the model without attack
        tf.print("\nEvaluating model without attack:")
        evaluate_model(model)

        evaluate_model_with_plots(model)
        plt.savefig('voltage_without_attack.png')
        

        # Evaluate the model with best attack if available
        if best_attack is not None:
            tf.print("\nEvaluating model with best attack:")
            t_eval = tf.random.uniform((1, 1), minval=0, maxval=TOTAL_TIME)
            evaluate_attack(model, best_attack, t_eval, Y_bus_tf, bus_data)
            evaluate_model(model, best_attack)
            plot_attack_vector(best_attack)
            evaluate_model_with_plots(model, best_attack)
            plt.savefig('voltage_with_attack.png')
        
        else:
            tf.print("\nNo valid attack vector was found during training.")

        plt.show()
            
    except Exception as e:
        tf.print(f"An error occurred during execution: {str(e)}")


