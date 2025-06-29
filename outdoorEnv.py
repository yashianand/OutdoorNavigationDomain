import numpy as np

'''
State representation: (x, y, puddle)
'''

class OutdoorMDP:
    def __init__(self, grid, is_oracle=False):
        self.grid = grid = np.asarray(grid, dtype='c')
        self.grid_list = [[c.decode('utf-8') for c in line] for line in self.grid.tolist()]
        self.actions = [0, 1, 2, 3] # left, up, right, down
        self.n_actions = len(self.actions)
        self.state = None
        self.terminal_state = None
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_states = self.rows * self.cols
        self.num_walls = sum(row.count('#') for row in self.grid_list)
        self.num_states_not_wall = self.num_states - sum(row.count('#') for row in self.grid_list)
        self.oracle_demos = {}
        self.oracle_q_values = {}
        self.agent_q_values = {}
        self.is_oracle = is_oracle
        self.get_all_reward = False
        self.all_states = None
        self.domain = 'outdoor'
        self.start_state = (np.where(self.grid == b'A')[0].item(), np.where(self.grid == b'A')[1].item(), False)
        self.rewards = []
        self.theta = []
        self.reward_fn = 'tabular'
        self.n_state_features = len(['step_reward', 'nse', 'goal'])  # step_reward, nse, goal
        self.reset()

    def reset(self, full_grid=True):
        if full_grid:
            self.oracle_demos = {}
            self.oracle_q_values = {}
            self.agent_q_values = {}
            all_states = self.getStateFactorRep()
        self.terminal_state = (np.where(self.grid == b'G')[0].item(), np.where(self.grid == b'G')[1].item(), False)
        self.state = self.start_state
        self.all_states = self.getStateFactorRep()
        self.n_states = len(self.all_states)
        self.theta = []
        self.state_to_index = {state: i for i, state in enumerate(self.all_states)}

    def getStateFactorRep(self):
        featureRep = []
        for i in range(self.rows):
            for j in range(self.cols):
                currState = self.grid_list[i][j]
                if currState != "#":
                    if currState == 'P':
                        featureRep.append((i, j, True))
                    else:
                        featureRep.append((i, j, False))
        return featureRep

    def step(self, action, evaluate=False):
        terminal = False
        successors, succ_probabilities = self.get_successors(self.state, action)
        next_state_idx = np.random.choice(len(successors), p=succ_probabilities)
        self.state = successors[next_state_idx]
        reward = self.get_reward(self.state, action, is_policy_iter=True)
        if self.is_goal(self.state):
            terminal = True
        return successors[next_state_idx], reward, succ_probabilities[next_state_idx], terminal

    def getActionFactorRep(self, a):
        if a == 0: # left
            return (0,-1)
        elif a == 1: # up
            return (-1,0)
        elif a == 2: # right
            return (0,1)
        else: # down
            return (1,0)

    def get_actions(self, state):
        return [0, 1, 2, 3]

    def is_boundary(self, state):
        x, y = state
        return (x <= 0 or x >= self.rows-1 or y <= 0 or y >= self.cols-1 )

    def is_goal(self, state):
        return state == self.terminal_state

    def move(self, currFactoredState, action):
        x, y, puddle = currFactoredState
        new_state = tuple(x + y for (x, y) in zip((x, y), self.getActionFactorRep(action)))
        if self.is_boundary(new_state):
            return currFactoredState, True
        else:
            if self.grid_list[new_state[0]][new_state[1]] == 'P':
                return (new_state[0], new_state[1], True), False
            else:
                return (new_state[0], new_state[1], False), False

    def get_side_states(self, state, action):
        side_states =[]
        for a in range(self.n_actions):
            if a != action:
                new_state, is_wall = self.move(state, a)
                if not is_wall:
                    side_states.append(new_state)
                elif is_wall:
                    side_states.append(state)
        return side_states

    def get_transition(self, curr_state, action, next_state):
        succ_factored_state, is_wall = self.move(curr_state, action)
        sstates = self.get_side_states(curr_state, action)

        success_prob = 0.95
        fail_prob = 0.05/3

        if is_wall:
            # print("hit boundary")
            transition_probs = []
            for feature_idx in range(len(curr_state)):
                if (curr_state[feature_idx] == next_state[feature_idx]):
                    transition_probs.append(1)
                else:
                    transition_probs.append(0)
            return np.prod(transition_probs)

        elif not is_wall:
            # print("no boundary")
            transition_probs = []
            if ((next_state[0], next_state[1])==(succ_factored_state[0], succ_factored_state[1])):
                transition_probs.append(success_prob)
                if (next_state[2]==succ_factored_state[2]):
                    transition_probs.append(1)
                elif (next_state[2]!=succ_factored_state[2]):
                    transition_probs.append(0)
                return np.prod(transition_probs)

            for side_state in sstates:
                if ((next_state[0], next_state[1])==(side_state[0], side_state[1])):
                    # print(sstates)
                    # print("if condn {}".format(side_state))
                    state_count = sstates.count(next_state)
                    fail_prob *= state_count
                    transition_probs.append(fail_prob)
                    if (next_state[2]==side_state[2]):
                        transition_probs.append(1)
                    elif (next_state[2]!=side_state[2]):
                        transition_probs.append(0)
                    return np.prod(transition_probs)

        return 0

    def get_possible_next_states(self, state):
        possible_states = set()
        for action in range(self.n_actions):
            next_state, _ = self.move(state, action)

            possible_states.add(next_state)
        return possible_states

    def get_successors(self, state, action):
        successors, succ_probabilities = [], []
        for next_state in self.get_possible_next_states(state):
            p = self.get_transition(state, action, next_state)
            if p > 0:
                successors.append(next_state)
                succ_probabilities.append(p)
        return successors, succ_probabilities

    def linear_reward(self, state):
        is_goal = self.is_goal(state)
        if is_goal:
            feature_vector = np.array([1, 0, 1])
        else:
            x, y, puddle = state
            is_puddle = 1 if puddle else 0
            feature_vector = np.array([1, is_puddle, 0])
        return np.dot(self.theta, feature_vector)

    def get_reward(self, state, action, is_policy_iter=False):
        state_reward = None
        x,y, puddle = state
        if is_policy_iter:
            goal = 100
            nse = -10
            step_reward = -1

        if self.reward_fn=='linear':
            if self.is_oracle:
                self.theta = [step_reward, nse, goal]
            elif self.is_oracle==False and len(self.theta)==0:
                self.theta = [step_reward, 0, goal]
            return self.linear_reward(state)

        elif self.reward_fn=='tabular':
            if self.is_goal(state) == True:
                return goal
            if self.is_oracle or self.get_all_reward:
                if puddle==1:
                    return nse
                else:
                    return step_reward
            else:
                return step_reward
        return None