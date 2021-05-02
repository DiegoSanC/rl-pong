import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math
from IPython.display import clear_output

class Environment:
    def __init__(self, max_life=3, height_px=40, width_px=50, mov_px=3):
        """

        :param max_life: num max lives
        :param height_px: height of environment matrix
        :param width_px: width of environment matrix
        :param mov_px: move of agent and ball in pixels
        """

        self.action_space = ['Up', 'Down']

        self._step_penalization = 0

        #The agente only makes movs in vertical axys. In each of those vertical positions,
        # there is going to be a one 2D table for the ball position
        self.state = [0, 0, 0]

        self.total_reward = 0

        # 3pixels movements for the ball and the agent
        self.dx = mov_px
        self.dy = mov_px

        rows = math.ceil(height_px / mov_px)
        columns = math.ceil(width_px / mov_px)


        self.positions_space = np.array([[[0 for z in range(columns)]
                                          for y in range(rows)]
                                         for x in range(rows)])

        self.lives = max_life
        self.max_life = max_life

        self.x_ball = random.randint(int(width_px / 2), width_px)
        self.y_ball = random.randint(0, height_px - 10)

        self.player_height = int(height_px / 4)

        self.player_1 = self.player_height  # posic. inicial del player

        self.score = 0

        self.width_px = width_px
        self.height_px = height_px
        self.radius = 2.5

    def reset(self):
        """
        Reset environment
        :return:
        """
        self.total_reward = 0
        self.state = [0, 0, 0]
        self.lives = self.max_life
        self.score = 0
        self.x_ball = random.randint(int(self.width_px / 2), self.width_px)
        self.y_ball = random.randint(0, self.height_px - 10)
        return self.state

    def step(self, action, animate=False):
        """
        Execute apply_action, check if agent has lost all lives, computes reward as sum of score and step_penalization
        and updates total_reward with new calculated reward
        :param action:
        :param animate:
        :return:
        """
        self.apply_action(action, animate)
        done = self.lives <= 0
        reward = self.score
        reward += self._step_penalization
        self.total_reward += reward
        return self.state, reward, done

    def apply_action(self, action, animate=False):
        """
        Apply one of two possible actions comming from the agent. In this case, actions can be Up an Down, and
        both are represented as a movement of x num of pixels in the board (negative or positive)
        :param action:
        :param animate:
        :return:
        """
        if action == "Up":
            self.player_1 += abs(self.dy)
        elif action == "Down":
            self.player_1 -= abs(self.dy)

        #We represent the movement of th agent
        self.step_player()

        #We represent the movement of the ball and the possible collision that will determine
        # if the agent gets a reward or a punishment
        self.step_ball()

        if animate:
            clear_output(wait=True)
            fig = self.draw_frame()
            plt.show()

        #We set new state after agent and ball have moved
        self.state = [math.floor(self.player_1 / abs(self.dy)) - 2, math.floor(self.y_ball / abs(self.dy)) - 2,
                      math.floor(self.x_ball / abs(self.dx)) - 2]

    def detect_collision(self, ball_y, player_y):
        """
        Detects if there is a collision between agent and ball. This method is called if in x coor agent and ball
        are close enough
        :param ball_y: y coor of ball
        :param player_y: y coor agent
        :return:
        """
        #If the top of the player is higher than y-bottom of the ball is a collision.
        # If bottom of the agent is lower than y-top of the ball, then whe have collision.
        # when ball hits in any other point of the agent, both conditions are going to be matched
        if (player_y+self.player_height >= (ball_y-self.radius)) and (player_y <= (ball_y+self.radius)):
            return True
        else:
            return False

    def step_player(self):
        """
        computes an agent step
        :return:
        """
        #Upper limit
        if self.player_1 + self.player_height >= self.height_px:
            self.player_1 = self.height_px - self.player_height
        elif self.player_1 <= -abs(self.dy):
            self.player_1 = -abs(self.dy)

    def step_ball(self):
        """
        Computes a ball step in the matrix
        :return:
        """
        self.x_ball += self.dx
        self.y_ball += self.dy

        #the ball bounces so the sense of the x movements is inverted
        if self.x_ball <= 3 or self.x_ball > self.width_px:
            self.dx = -self.dx

            #As the radius is 2.5 we have to evaluate if there is a collision
            if self.x_ball <= 3:
                ret = self.detect_collision(self.y_ball, self.player_1)

                if ret:
                    self.score = 10
                else:
                    self.score = -10
                    self.lives -= 1
                    if self.lives > 0:
                        self.x_ball = random.randint(int(self.width_px / 2), self.width_px)
                        self.y_ball = random.randint(0, self.height_px - 10)
                        self.dx = abs(self.dx)
                        self.dy = abs(self.dy)
        else:
            self.score = 0

        # the ball bounces so the sense of the y movements is inverted
        if self.y_ball < 0 or self.y_ball > self.height_px:
            self.dy = -self.dy

    def draw_frame(self):
        """
        For drawing environment and movements of agent a ball
        :return:
        """
        fig = plt.figure(figsize=(5, 4))
        a1 = plt.gca()
        circle = plt.Circle((self.x_ball, self.y_ball), self.radius, fc='slategray', ec="black")
        a1.set_ylim(-5, self.height_px + 5)
        a1.set_xlim(-5, self.width_px + 5)

        rectangle = plt.Rectangle((-5, self.player_1), 5, self.player_height, fc='gold', ec="none")
        a1.add_patch(circle)
        a1.add_patch(rectangle)
        # a1.set_yticklabels([]);a1.set_xticklabels([]);
        plt.text(4, self.height_px, "SCORE:" + str(self.total_reward) + "  LIFE:" + str(self.lives), fontsize=12)
        if self.lives <=0:
            plt.text(10, self.height_px - 14, "GAME OVER", fontsize=16)
        elif self.total_reward >= 1000:
            plt.text(10, self.height_px - 14, "YOU WIN!", fontsize=16)
        return fig



