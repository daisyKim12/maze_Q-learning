import numpy as np
import matplotlib.pyplot as plt

class Optimization:
    def __init__(self, maze):
        self.maze = maze

    def visualize_q_table(self, q_table):
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.axis('tight')
        rows, cols = self.maze.num_rows, self.maze.num_cols
        cell_text = []
        cell_colors = []

        for col in range(cols-1, -1, -1):
            row_text = []
            row_colors = []
            for row in range(rows):
                state = (col, row) 
                if not self.maze.is_valid_coor(state):
                    row_text.append('')
                    row_colors.append('white')
                    continue
                actions = q_table[state]
                action_values = []
                for action in ['up', 'down', 'left', 'right']:
                    if action in actions:
                        if action == 'up':
                            label = 'U:'
                        elif action == 'down':
                            label = 'D:'
                        elif action == 'left':
                            label = 'L:'
                        elif action == 'right':
                            label = 'R:'
                        action_values.append(f'{label}{actions[action]:.2f}')
                    else:
                        action_values.append('')
                row_text.append('\n'.join(action_values))
                
                # Label start and end coordinates
                if state == self.maze.entry_coor:
                    row_colors.append('green')
                elif state == self.maze.exit_coor:
                    row_colors.append('red')
                else:
                    row_colors.append('white')

            cell_text.append(row_text)
            cell_colors.append(row_colors)
        
        table = ax.table(cellText=cell_text, cellColours=cell_colors, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        plt.show()
