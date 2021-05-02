import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
from IPython.display import clear_output

class Agent:

    def __init__(self, game, policy=None, discount_factor=0.1, learning_rate=0.1, exploitation_rate=0.9):

        """

        :param game:
        :param policy:
        :param discount_factor:
        :param learning_rate:
        :param ratio_explotacion:
        """

        # Build policy table
        if policy is not None:
            self._q_table = policy
        else:
            position = list(game.positions_space.shape)
            position.append(len(game.action_space))
            self._q_table = np.zeros(position)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.exploitation_rate = exploitation_rate

    def get_next_step(self, state, game):
        """
        Select next best action (max) with the information given by the state. Some times, it generates a random
        move (up or down) to explore new possible combinations

        :param state: current state (agent_position, ball_x, ball_y)
        :param game: environment
        :return:
        """
        # Random step for exploring
        next_step = np.random.choice(list(game.action_space))

        # np.random.uniform returns a number between [0,1]
        if np.random.uniform() <= self.exploitation_rate:
            # in the current state we select the max value between 2 possible actions (Up, Down)
            idx_action = np.random.choice(np.flatnonzero(
                self._q_table[state[0], state[1], state[2]] == self._q_table[state[0], state[1], state[2]].max()
            ))
            next_step = list(game.action_space)[idx_action]

        return next_step

    # We update policy with obtained rewards
    def update(self, game, old_state, action_taken, reward_action_taken, new_state, reached_end):
        """
        Computes updates of the policy tables
        :param game: environment
        :param old_state: last known state before agent action taken
        :param action_taken: next action taken by the agent
        :param reward_action_taken: reward obtained by the action (10 when colission, -10 when ball hits left wall, 0
        otherwise)
        :param new_state: current state after agent has taken an action
        :param reached_end: if game has ended
        :return:
        """
        #we get the index (0 or 1) depending on the action selected in get_next_step
        idx_action_taken = list(game.action_space).index(action_taken)


        #for agent and ball positions we get 2 possible options (up and down) which initially have two 0s
        current_q_value_options = self._q_table[old_state[0], old_state[1], old_state[2]]
        #This change the position of the agent so we need the table for that new agent position
        current_q_value = current_q_value_options[idx_action_taken]

        future_q_value_options = self._q_table[new_state[0], new_state[1], new_state[2]]

        # future_max_q_value = R + (lambda * maxQ(s'))
        future_max_q_value = reward_action_taken + self.discount_factor * future_q_value_options.max()
        if reached_end:
            future_max_q_value = reward_action_taken  # maximum reward

        # Q^(s,a) = Q(s,a) + alpha*[future_max_q_value - Q(s,a)
        self._q_table[old_state[0], old_state[1], old_state[2], idx_action_taken] = current_q_value + \
                                                                                    self.learning_rate * \
                                                                                    (future_max_q_value -
                                                                                     current_q_value)

    def print_policy(self):
        for row in np.round(self._q_table, 1):
            for column in row:
                print('[', end='')
                for value in column:
                    print(str(value).zfill(5), end=' ')
                print('] ', end='')
            print('')

    def get_policy(self):
        return self._q_table