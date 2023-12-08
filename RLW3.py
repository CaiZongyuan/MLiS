import numpy as np

# Define the gridworld rewards
r = np.zeros((5, 5))  # create a 5x5 matrix, index 0 is unused
r[1, 4] = 6
r[1, 3] = 5
r[1, 2] = 4
r[1, 1] = 3
r[2, 4] = 5
r[2, 3] = 4
r[2, 2] = 3
r[2, 1] = 3
r[3, 4] = 4
r[3, 3] = 3
r[3, 2] = 2
r[3, 1] = 1
r[4, 4] = 3
r[4, 3] = 2
r[4, 2] = 1
r[4, 1] = 0

# Define the state-action value function Q_psi
def Q_psi(state, action, psi_1, psi_2):
    x, y = state
    delta_x = 1 if action == 2 else -1 if action == 4 else 0
    delta_y = 1 if action == 1 else -1 if action == 3 else 0
    return 6 - (x + delta_x - psi_1) ** 2 - (y + delta_y - psi_2) ** 2

# Function to calculate the next state based on current state and action
def next_state(current_state, action):
    x, y = current_state
    if action == 1 and y < 4:  # Up
        y += 1
    elif action == 2 and x < 4:  # Right
        x += 1
    elif action == 3 and y > 1:  # Down
        y -= 1
    elif action == 4 and x > 1:  # Left
        x -= 1
    return [x, y]

# Function to compute the partial derivatives of Q_psi
def partial_derivatives_Q_psi(state, action, psi_1, psi_2):
    x, y = state
    delta_x = 1 if action == 2 else -1 if action == 4 else 0
    delta_y = 1 if action == 1 else -1 if action == 3 else 0
    partial_psi_1 = 2 * (x + delta_x - psi_1)
    partial_psi_2 = 2 * (y + delta_y - psi_2)
    return partial_psi_1, partial_psi_2

# Calculate delta_Q
def delta_Q(state, action, psi_1, psi_2):
    q_current = Q_psi(state, action, psi_1, psi_2)
    new_state = next_state(state, action)
    q_new_state = [Q_psi(new_state, a, psi_1, psi_2) for a in range(1, 5)]
    max_q_new_state = max(q_new_state)
    delta_q = max_q_new_state + r[new_state[0], new_state[1]] - q_current
    return delta_q

# Initial parameters
psi_1 = 3
psi_2 = 1

# State-action pairs and actions for gradient calculation
state_action_pairs = [([3, 1], 2), ([2, 4], 3)]

Q_psi1 = Q_psi(state_action_pairs[0][0], state_action_pairs[0][1], psi_1, psi_2)
Q_psi2 = Q_psi(state_action_pairs[1][0], state_action_pairs[1][1], psi_1, psi_2)

print("Q_psi1: ", Q_psi1, "Q_psi2: ", Q_psi2)

# Calculate the gradients and update the parameters
grad_psi_1 = grad_psi_2 = 0
learning_rate = 0.05
for (state, action) in state_action_pairs:
    delta_q = delta_Q(state, action, psi_1, psi_2)
    print("delta_q: ", delta_q)
    partial_1, partial_2 = partial_derivatives_Q_psi(state, action, psi_1, psi_2)
    grad_psi_1 += delta_q * partial_1
    grad_psi_2 += delta_q * partial_2
    print("partial_1: ", partial_1, "partial_2: ", partial_2)
 
# Averaging the gradients
num_transitions = len(state_action_pairs)
grad_psi_1_avg = -grad_psi_1 / num_transitions
grad_psi_2_avg = -grad_psi_2 / num_transitions

print("grad_psi_1_avg: ", grad_psi_1_avg, "grad_psi_2_avg: ", grad_psi_2_avg)

# Update parameters
psi_1_updated = psi_1 - learning_rate * grad_psi_1_avg
psi_2_updated = psi_2 - learning_rate * grad_psi_2_avg

print("psi_1_updated: ", psi_1_updated, "psi_2_updated: ", psi_2_updated)

