import numpy as np

def read_envs(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    array_from_file = eval(data)

    return array_from_file

def print_values(vals, env_size, pi=True):
    action_symbols = {0: '<', 1: '^', 2: '>', 3: 'v'}  # Define symbols for each action
    for i in range(env_size[0]):  # Iterate over rows
        row = ''
        for j in range(env_size[1]):  # Iterate over columns
            state_index = i * env_size[1] + j  # Calculate the state index
            if pi==True:
                action = np.argmax(vals[state_index])
                row += action_symbols[action] + ' '  # Append the action symbol to the row string
            else:
                row += str(round(vals[state_index])) + ' '
        print(row)

def simulate_trajectory(grid, policy):
    trajectory = []
    trial_reward = 0
    terminal= False
    grid.reset(full_grid=False)
    action = np.argmax(policy[grid.state_to_index[grid.state]])
    trajectory.append([tuple(grid.state), action])
    observation, reward, _, terminal = grid.step(action)
    action = np.argmax(policy[grid.state_to_index[observation]])
    trajectory.append([tuple(observation), action])
    trial_reward += reward
    while not terminal:
        observation, reward, _, terminal = grid.step(action)
        action = np.argmax(policy[grid.state_to_index[observation]])
        trajectory.append([tuple(observation), action])
        trial_reward += reward
    return trial_reward, trajectory