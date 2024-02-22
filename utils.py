import matplotlib.pyplot as plt


def plot_errors(minority_probs, all_errors, ldp_errors, e0, e1, times, n):
    colors = plt.cm.tab10(range(len(minority_probs)))
    for minority_prob, errors, ldp_error, color in zip(minority_probs, all_errors, ldp_errors, colors):


        # Plot the line based on array a
        label = f'Minority: {minority_prob, }; final error: {errors[-1]:.00005}'
        plt.plot(range(1,len(errors)+1), errors, marker='o', color=color, label=label, markersize=5)

        # Mark a star where y-axis corresponds to v
        # Mark a star where y-axis corresponds to v
        ldp_label = f'Error: {ldp_error:.000005}'
        plt.scatter(1, ldp_error, color=color, marker='*', s=20, label=ldp_label)


    # Add labels and legend
    plt.xlabel('Interactions')
    plt.ylabel('MSE')
    plt.title(f'eo_{e0:.2f}_e1_{e1:.2f}_times_{times}_{n}.png')
    plt.legend(prop={'size': 6})

    # Show the plot
    plt.grid(True)
    plt.savefig(f'exp_baseline/epso_{e0:.2f}_e1_{e1:.2f}_times_{times}_n_{n}.png')
