import matplotlib.pyplot as plt


def plot_errors(minority_probs, all_errors, ldp_errors, baseline_errors, e0, times, n, main_folder):
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
    plt.ylabel('Majority Count Difference')
    plt.title(f'eo_{e0:.2f}_times_{times}_{n}.png')
    plt.legend(prop={'size': 6})

    # Show the plot
    plt.grid(True)
    plt.savefig(f'{main_folder}/epso_{e0:.2f}_times_{times}_n_{n}.png')


def calculate_error(real, est,n):
    percentage_change = (abs(est-real))
    return percentage_change[0]*n


def calculate_epsilons_for_interactions(total, first, eps_inf, threshold=0.1):
    result = [first]
    current_value = first
    remaining = total - first
    # Iterate until the current value is greater than 0.6
    while True:
        current_value = 0.1 * remaining  # Decrease the current value by 60%
        
        if current_value + sum(result) > total:
            result.append(current_value + remaining)
            break
        elif current_value < threshold:
            if threshold + sum(result) > total:
                result[-1] = result[-1] + remaining
                break
            else:
                result.append(threshold)
                remaining -= threshold
                continue
        remaining -= current_value
        result.append(current_value)
        
        
    # print(remaining)
    # if remaining > 0:
    #     result.append(remaining)
    result.sort(reverse=True)
    return [[eps_inf, result[i]] for i in range(len(result))]



