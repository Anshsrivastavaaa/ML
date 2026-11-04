# ============================================================
# 11.py — Q-Learning (Reinforcement Learning)
#          Training an agent to navigate a 4×4 Grid World
# ============================================================

# --- Import libraries ---
import numpy as np     # For numerical operations (Q-table, argmax)
import random          # For random number generation (exploration)

# --- Define the Grid World environment ---
# A 4×4 grid with 16 states (numbered 0–15):
#   [ 0][ 1][ 2][ 3]
#   [ 4][ 5][ 6][ 7]
#   [ 8][ 9][10][11]
#   [12][13][14][15]  ← state 15 is the GOAL
states = 16
actions = 4   # 4 possible actions: 0=Up, 1=Down, 2=Left, 3=Right
goal = 15     # The target state the agent must reach

# --- Initialize the Q-table ---
# Q-table has shape (16 states × 4 actions), initialized to zeros
# Each entry Q[s, a] represents the expected future reward for taking action 'a' in state 's'
Q = np.zeros((states, actions))

# --- Hyperparameters ---
alpha = 0.1      # Learning rate: how much new information overrides old Q-values (0–1)
gamma = 0.9      # Discount factor: how much the agent values future vs immediate rewards (0–1)
epsilon = 0.1    # Exploration rate: probability of taking a random action instead of the best known
episodes = 1000  # Number of training episodes (full games from start to goal)

# --- State transition function ---
def get_next_state(state, action):
    """
    Given a current state and action, returns the next state.
    The agent cannot move outside the 4×4 grid boundaries.
    If the action would take it out of bounds, it stays in the same state.
    """
    # Convert flat state index to (row, col) coordinates
    row, col = divmod(state, 4)

    # Apply the action with boundary checks
    if action == 0 and row > 0:      # Up: move one row up (if not at top edge)
        row -= 1
    elif action == 1 and row < 3:    # Down: move one row down (if not at bottom edge)
        row += 1
    elif action == 2 and col > 0:    # Left: move one column left (if not at left edge)
        col -= 1
    elif action == 3 and col < 3:    # Right: move one column right (if not at right edge)
        col += 1

    # Convert (row, col) back to flat state index
    return row * 4 + col

# --- Q-Learning training loop ---
for episode in range(episodes):
    # Start each episode at a random state (0–14), excluding the goal state
    state = random.randint(0, 14)

    # Each episode runs for at most 100 steps
    for step in range(100):
        # --- Epsilon-greedy action selection ---
        # With probability epsilon: EXPLORE (choose a random action)
        # With probability (1-epsilon): EXPLOIT (choose the best known action)
        if random.random() < epsilon:
            action = random.randint(0, 3)           # Random exploration
        else:
            action = np.argmax(Q[state])             # Greedy exploitation

        # Execute the action and observe the next state
        next_state = get_next_state(state, action)

        # --- Compute the reward ---
        # +10 for reaching the goal, -1 for every other step (encourages shortest path)
        reward = 10 if next_state == goal else -1

        # --- Update Q-value using the Bellman equation ---
        # Q(s, a) ← Q(s, a) + α * [reward + γ * max_a' Q(s', a') - Q(s, a)]
        # This moves Q(s, a) toward the observed reward + discounted future value
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state])
            - Q[state, action]
        )

        # Move to the next state
        state = next_state

        # If the agent reached the goal, end this episode
        if state == goal:
            break

# --- Display the learned Q-table ---
# Each row shows Q-values for [Up, Down, Left, Right] actions in that state
print("Learned Q-Table:")
print(np.round(Q, 2))

# --- Display the optimal policy ---
# For each state, show the direction the agent should move (highest Q-value action)
arrows = ['U', 'D', 'L', 'R']  # Mapping: 0=Up, 1=Down, 2=Left, 3=Right

print("\nOptimal Policy:")
for state in range(states):
    if state == goal:
        print(" G ", end=" ")             # Mark the goal state with 'G'
    else:
        print(arrows[np.argmax(Q[state])], end="  ")  # Print the best action for this state

    # Print a newline after every 4 states (end of a grid row)
    if state % 4 == 3:
        print()
