import numpy as np
import copy

def compute_q_for_pi(env, pi, gamma):
    state_values = policy_evaluation(env, pi, gamma)
    return __compute_q_with_v(env, state_values, gamma)

def __compute_q_with_v(env, state_values, gamma):
    all_states = env.all_states
    state_to_index = env.state_to_index
    q_values = np.concatenate([np.expand_dims(__compute_q_s_with_v(env, s, all_states, state_to_index, state_values, gamma), axis=0)
                               for s in range(env.n_states)])
    return q_values


def __compute_q_s_with_v(env, s, all_states, state_to_index, state_values, gamma):
    state = all_states[s]
    q_values_s = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        if env.reward_fn=='tabular':
            r = env.rewards[s]
        elif env.reward_fn=='linear':
            r = env.get_reward(state, a, is_policy_iter=True)
        if env.is_goal(state):
            q_values_s[a] = r
            continue
        successors, succ_probabilities = env.get_successors(state, a)
        # q_values_s[a] = r + gamma * sum(succ_probabilities[i] * state_values[state_to_index[successors[i]]] for i in range(len(successors)))
        q_values_s[a] = sum([succ_probabilities[i] * (r + gamma * state_values[state_to_index[successors[i]]]) for i in range(len(successors))])
    return q_values_s


def policy_evaluation(env, pi, gamma, state_values_for_pi=None, delta=1e-4):
    if state_values_for_pi is None:
        state_values_for_pi = np.zeros(len(env.all_states))

    all_states = env.all_states
    state_to_index = env.state_to_index
    gamma_state_values = gamma * state_values_for_pi  # Precompute for efficiency

    if env.reward_fn == 'tabular':
        rewards = env.rewards
    else:
        rewards = np.zeros(len(env.all_states))

    while True:
        max_delta = 0
        for s, state in enumerate(all_states):
            old_vs = state_values_for_pi[s]
            action = np.argmax(pi[s])

            if env.reward_fn == 'linear':
                if env.is_goal(state):
                    rewards[s] = env.get_reward(state, action, is_policy_iter=True)
                    continue
                rewards[s] = env.get_reward(state, action, is_policy_iter=True)

            successors, succ_probabilities = env.get_successors(state, action)
            successor_indices = [state_to_index[succ] for succ in successors]
            new_vs = np.dot(succ_probabilities, rewards[s] + gamma_state_values[successor_indices])

            state_values_for_pi[s] = new_vs
            max_delta = max(abs(new_vs - old_vs), max_delta)
        # gamma_state_values = gamma * state_values_for_pi

        if max_delta < delta:
            break

    return state_values_for_pi


def policy_iteration(env, gamma=0.95, pi=None):
    if pi is None:
        pi = np.zeros((len(env.all_states), len(env.actions)))
    n_iter = 0
    state_values = np.zeros(len(env.all_states))
    while True:
        old_pi = copy.deepcopy(pi)
        state_values = policy_evaluation(env, pi, gamma, state_values)
        # print(state_values)
        # input()
        q_values = __compute_q_with_v(env, state_values, gamma)
        pi = np.zeros_like(pi)
        pi[np.arange(len(env.all_states)), np.argmax(q_values, axis=1)] = 1
        if np.all(old_pi == pi):
            return pi
        else:
            n_iter += 1

def compute_q_via_dp(env, delta=1e-4, gamma=0.99):
    state_values = np.zeros(env.n_states, dtype=np.float32)
    while True:
        max_delta = 0
        for s in range(env.n_states):
            old_vs = state_values[s]
            state_values[s] = np.max(__compute_q_s_with_v(env, s, state_values, gamma))
            max_delta = max(abs(state_values[s] - old_vs), max_delta)
        # print('max delta: {:>.5f}'.format(max_delta))
        if max_delta < delta:
            break
    return __compute_q_with_v(env, state_values, gamma)
