import numpy as np
import random

# Grid World: 4 x 4
states = 16
actions = 4  # 0=Up, 1=Down, 2=Left, 3=Right
goal = 15

# Q-table
Q = np.zeros((states, actions))

# Parameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

# Get next state
def get_next_state(state, action):
    row, col = divmod(state, 4)

    if action == 0 and row > 0:
        row -= 1
    elif action == 1 and row < 3:
        row += 1
    elif action == 2 and col > 0:
        col -= 1
    elif action == 3 and col < 3:
        col += 1

    return row * 4 + col

# Q-Learning algorithm
for episode in range(episodes):
    state = random.randint(0, 14)

    for step in range(100):
        # Exploration or exploitation
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[state])

        next_state = get_next_state(state, action)

        # Reward
        reward = 10 if next_state == goal else -1

        # Update Q-value
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state])
            - Q[state, action]
        )

        state = next_state

        if state == goal:
            break

# Display Q-table
print("Learned Q-Table:")
print(np.round(Q, 2))

# Display optimal policy
arrows = ['U', 'D', 'L', 'R']

print("\nOptimal Policy:")
for state in range(states):
    if state == goal:
        print(" G ", end=" ")
    else:
        print(arrows[np.argmax(Q[state])], end="  ")

    if state % 4 == 3:
        print()

 
