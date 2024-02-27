import matplotlib.pyplot as plt


def plot_errors(minority_probs, all_errors, ldp_errors, baseline_errors, e0, e1, times, n, main_folder):
    colors = plt.cm.tab10(range(len(minority_probs)))
    for minority_prob, errors, ldp_error, baseline_error, color in zip(minority_probs, all_errors, ldp_errors,baseline_errors, colors):


        # Plot the line based on array a
        label = f'Minority: {minority_prob, }; final error: {errors[-1]:.00005}'
        plt.plot(range(1,len(errors)+1), errors, marker='o', color=color, label=label, markersize=5)

        # Mark a star where y-axis corresponds to v
        # Mark a star where y-axis corresponds to v
        ldp_label = f'Error: {ldp_error:.000005}'
        plt.scatter(1, ldp_error, color=color, marker='*', s=20, label=ldp_label)

        baseline_label = f'B Error: {baseline_error:.000005}'
        plt.scatter(1, baseline_error, color=color, marker='.', s=20, label=baseline_label)


    # Add labels and legend
    plt.xlabel('Interactions')
    plt.ylabel('MSE')
    plt.title(f'eo_{e0:.2f}_e1_{e1:.2f}_times_{times}_{n}.png')
    plt.legend(prop={'size': 6})

    # Show the plot
    plt.grid(True)
    plt.savefig(f'{main_folder}/epso_{e0:.2f}_e1_{e1:.2f}_times_{times}_n_{n}.png')


def calculate_error(real, est):
    percentage_change = (abs(est-real)/real) * 100
    return percentage_change[0]


def calculate_epsilons_for_interactions(total, first, eps_inf, threshold=0.2):
    result = [first]
    current_value = first
    remaining = total - first
    threshold = 0.1
    # Iterate until the current value is greater than 0.6
    while True:
        current_value *= 0.6  # Decrease the current value by 60%
        remaining -= current_value
        if current_value + sum(result) > total:
            result.append(current_value + remaining)
            break
        elif current_value < threshold:
            result[-1] = result[-1] + current_value
            break
        result.append(current_value)
        
        
    # print(remaining)
    # if remaining > 0:
    #     result.append(remaining)
    
    return [[eps_inf, result[i]] for i in range(len(result))]



