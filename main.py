from maze_manager import MazeManager
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    manager = MazeManager()

    parameters = {
        'episodes': 1000,
        'alpha': 0.05,
        'gamma': 0.7,
        'epsilon': 0.95,
        'invalid_reward': -10
    }

    print("\nStart Q-learning 3 times...")
    manager.show_q_table = True

    for i in range(1,3):     
        maze = manager.add_maze(20, 20)
        manager.show_maze(maze.id)

        input("Press Enter to continue...")

        cost, time = manager.solve_maze(maze.id, "perfrom_q_learning", parameters)

        input("Press Enter to continue...")

        manager.show_solution(maze.id)

    print("\nStart optimizing parameter alpha and gamma...")

    # Define the values for alpha and gamma that you want to optimize
    alpha_values = [0.2, 0.4, 0.6, 0.8]
    gamma_values = [0.2, 0.4, 0.6, 0.8]

    cost_performance = []
    time_performance = []
    
    maze_optimize = manager.add_maze(20, 20)
    manager.show_maze(maze_optimize.id)
    manager.show_q_table = False
    
    input("Press Enter to continue...")
    
    print("\nOptimizing parameter alpha and gamma...")

    for alpha in alpha_values:
        for gamma in gamma_values:

            # Update the values in the parameters dictionary
            parameters['alpha'] = alpha
            parameters['gamma'] = gamma
            # parameters['invalid_reward'] = -1

            cost, time = manager.solve_maze(maze_optimize.id, "perfrom_q_learning", parameters)
            cost_performance.append(cost)
            time_performance.append(time)
    
    # Reshape the cost_performance and time_performance arrays
    cost_performance = np.array(cost_performance).reshape(len(alpha_values), len(gamma_values))
    time_performance = np.array(time_performance).reshape(len(alpha_values), len(gamma_values))

    # Create separate subplots for cost performance and time performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot cost performance
    for i in range(len(gamma_values)):
        ax1.plot(alpha_values, cost_performance[:, i], marker='o', label=f"Gamma = {gamma_values[i]}")

    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Cost Performance')
    ax1.set_title('Cost Performance vs Alpha')
    ax1.legend()

    # Plot time performance
    for i in range(len(alpha_values)):
        ax2.plot(gamma_values, time_performance[i, :], marker='o', label=f"Alpha = {alpha_values[i]}")

    ax2.set_xlabel('Gamma')
    ax2.set_ylabel('Time Performance')
    ax2.set_title('Time Performance vs Gamma')
    ax2.legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show the plots
    plt.show()