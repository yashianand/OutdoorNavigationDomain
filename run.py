from outdoorEnv import OutdoorMDP
from solvers import policy_iteration
from helper_functions import *

def main(trainGrid_file, reward_type='tabular'):
    train_envs = read_envs(trainGrid_file)
    env = train_envs[0]

    env = OutdoorMDP(env)
    env.reset()
    env.is_oracle = True
    env.reward_fn = reward_type
    if env.reward_fn=='tabular':
        for s in env.all_states:
            r = env.get_reward(s, 0, is_policy_iter=True)
            env.rewards.append(r if r!=0 else 10)

    pi = policy_iteration(env)

    print("Policy Iteration completed.")
    print("Final Policy:")
    print_values(pi, [15, 15], pi=True)

    # Simulate a trajectory using the learned policy
    trajectory_reward, trajectory = simulate_trajectory(env, pi)
    print("Simulated Trajectory:", trajectory[1]) # list of list of state, action pairs

if __name__ == "__main__":
    trainGrid_file = 'grids/train.txt'
    reward_type = 'tabular'  # or 'linear'
    main(trainGrid_file, reward_type)
