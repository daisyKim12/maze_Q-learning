import time
import random
from visualize import Optimization
import numpy as np
import matplotlib.pyplot as plt

class MyQLearning():

    '''intialize maze to search solution'''
    def __init__(self, maze):                                              
        self.maze = maze                                                # maze to solve
        # self.path = list()                                              # current path as algorithm is searching 
        # self.total_spead = 0                                            # total searching time
        # self.total_cost = 0                                             # total searching cost
        self.learning_time = 0                                            # learning time
        self.q_table = {}
        self.visualize = Optimization(maze)
        self.max_path = maze.num_rows * maze.num_cols

    '''
    alpha = factor that maintains q table value
    gamma = decay factor
    epsilon = more random exploration when close to 1
    '''

    def q_learning(self, maze, episodes, alpha, gamma, epsilon, invalid_reward, show):
        
        print("\nInitializing Q-table with all zeros...")

        # initialize Q-table with all zeros
        for i in range(maze.num_rows):
            for j in range(maze.num_cols):
                self.q_table[(i, j)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

        # self.visualize.visualize_q_table(self.q_table)
        
        # Q-learning algorithm
        print("\nSolving the maze with Q-learning...")
        start= time.perf_counter()                           #starting time of searching

        for episode in range(episodes):
            state = maze.entry_coor
            done = False
            temp_gamma = gamma
            while not done:
                # choose action based on epsilon-greedy policy
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(['up', 'down', 'left', 'right'])
                else:
                    action = max(self.q_table[state], key=self.q_table[state].get)

                # take action and observe new state and reward
                x, y = state
                if action == 'up':
                    new_state = (x, y-1)
                elif action == 'down':
                    new_state = (x, y+1)
                elif action == 'left':
                    new_state = (x-1, y)
                else:
                    new_state = (x+1, y)

                # get reward for the new state
                reward, is_wall = maze.get_reward(state, new_state, invalid_reward)

                # check if the new state is valid and terminal
                if maze.is_valid_coor(new_state):
                    done = maze.is_terminal_coor(new_state)
                else:
                    # Penalize invalid moves
                    reward = invalid_reward
                    done = False
                    new_state = state

                # update Q-table only if the new state is valid
                if maze.is_valid_coor(new_state):
                    old_value = self.q_table[state][action]
                    next_max = max(self.q_table[new_state].values())
                    new_value = (1 - alpha) * old_value + alpha * (reward + temp_gamma * next_max)
                    self.q_table[state][action] = new_value
                    
                    # temp_gamma = temp_gamma*0.999

                print(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, Done: {done}")

                state = new_state
                
                # if(is_wall == False):
                #     state = new_state
                # else:
                #     state = state

                if done:
                    end = time.perf_counter()
                    self.learning_time = end - start                   # save the total time learning
                    break               
                
        print(f"Episode {episode} completed")

        if(show == True):
            self.visualize.visualize_q_table(self.q_table)

        return self.q_table, self.learning_time
    
    def q_learning_path(self, maze, q_table):
        # Start at the beginning of the maze
        current_state = maze.entry_coor

        # Initialize path and cost variables
        path = [(current_state, False)]
        cost = 0

        # Loop until the goal is reached
        while current_state != maze.exit_coor:
            #
            #Choose the action with the highest Q-value for the current state
            possible_actions = list(q_table[current_state].keys())
            
            #possible_actions = self.filter_invalid_action(current_state, possible_actions)
            
            values = [q_table[current_state][a] for a in possible_actions]



            max_value = max(values)
            indices = [i for i in range(len(values)) if values[i] == max_value]
            action_index = random.choice(indices)
            action = possible_actions[action_index]
            print("Current state:", current_state, "Action chosen:", action)
            
            # Calculate the next state and cost
            x, y = current_state
            if action == 'up':
                next_state = (x, y-1)
            elif action == 'down':
                next_state = (x, y+1)
            elif action == 'left':
                next_state = (x-1, y)
            else:
                next_state = (x+1, y)
            step_cost = maze.grid[next_state[0]][next_state[1]]

            # Add the step cost to the total cost
            cost += 1

            # Add the next state to the path
            path.append((next_state, False))

            # Update the current state
            current_state = next_state

            if(cost > self.max_path):
                break

        # Return the path and cost
        return path, cost

    def filter_invalid_action(self, current_state, possible_actions):
        x, y = current_state
        neighbours = self.maze.find_neighbours(x, y)
        validate_neighbours = self.maze.validate_neighbours_solve(neighbours, x, y,
                                     self.maze.exit_coor[0], self.maze.exit_coor[1], "brute-force")

        filtered_actions = []    
        for next_coor in validate_neighbours:
            x_new, y_new = next_coor
            if y_new == y-1:
                filtered_actions.append('up')
            elif  y_new == y+1:
                filtered_actions.append('down')
            elif x_new == x-1:
                filtered_actions.append('left')
            else:
                filtered_actions.append('right')
        
        if len(filtered_actions) == 0:
            # Handle the case when no valid actions remain
            # You can raise an exception, return a default action, or take any other desired action
            pass
        
        return filtered_actions