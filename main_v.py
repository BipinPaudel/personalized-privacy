import pandas as pd
import numpy as np
from data import gen_binary_data
from protocols.grr import GRR_Client, GRR_Aggregator_MI, GRR_Aggregator_IBU
from protocols.L_GRR import L_GRR_Client, L_GRR_Aggregator_MI, L_GRR_Client_Interaction, L_GRR_Aggregator_IBU
from sklearn.metrics import mean_squared_error
from user import User
from server import Server
import matplotlib.pyplot as plt
# 3.3409322826759606e-05
# 7.073518299087083e-07
if __name__=='__main__':
    
    # number_of_users = [100, 1000, 10000]
    # number_of_users = [100000]
    # for n in number_of_users:
    #     accuracy = 0
        
    #     plt.clf()
    #     evolving_histogram_exp = []
    #     for exp_ins in range(10):
    #         data = gen_binary_data(n=n, one_prob=0.4)
    #         k = 2
    #         real_freq = np.unique(data, return_counts=True)[-1] / n
    #         num_seed = 1

            
    #         print(f'Real frequency {real_freq}')
    #         epsilons = [[2,0.1], [0.1, 0.4], [0.1,0.8],[0.1,1], [0.1,1], [0.1,1],[0.1,1],[0.1,1],[0.1,1.2],[0.1,1.4],[0.1,1.4],[0.1,1.4]]
            
    #         # epsilons = [[2,0.1], [0.1, 1.4]]
    #         times = len(epsilons)
    #         server = Server(epsilons, k)

    #         #permanent randomization phase
    #         eps, p1, q1 = server.get_permanent_randomization_configs()
    #         estimated_histogram = None
    #         average_histograms = []
    #         # evolving_histogram_seed = []
    #         # for seed in range(num_seed):
    #         evolving_histogram = []
    #         users = [User(v) for v in data['v']]
    #         permanent_values = [user.permanent_randomization(p1, q1) for user in users]
    #         for time in range(times):
    #             # print(f'This is time: {time}')
    #             if time == 0:
    #                 eps_perm, p1, q1, eps_1, p2, q2 = server.get_instantaneous_randomization_configs(time)
    #                 reports = []
    #                 for user in users:
    #                     report = user.instantaneous_randomization(p1, q1, p2, q2)
    #                     reports.append(report)
    #                 estimated_histogram = server.get_estimated_histogram(reports, time, private_list = np.array(data['v']))

    #             else:
    #                 eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj = server.get_instantaneous_randomization_configs(time)
    #                 majority = estimated_histogram.argmax()
    #                 minority = estimated_histogram.argmin()
    #                 reports = []
    #                 for u in users:
    #                     if u.private_value == majority:
    #                         report = u.instantaneous_randomization(p1, q1, p_maj, q_maj)
    #                     else:
    #                         report = u.instantaneous_randomization(p1, q1, p_min, q_min)
    #                     reports.append(report)
    #                 estimated_histogram = server.get_estimated_histogram(reports, time, private_list = np.array(data['v']))

    #             evolving_histogram.append(estimated_histogram)
    #         # evolving_histogram_seed.append(evolving_histogram)
            
    #         # evolving_histogram_seed = np.array(evolving_histogram_seed)
    #         evolving_histogram_exp.append(evolving_histogram)
    #     evolving_histogram_exp = np.array(evolving_histogram_exp)
    #     errors = []
    #     first_error = None
    #     for time in range(times):
    #         arr = evolving_histogram_exp[:,time]
    #         avg_hist = np.mean(arr, axis=0)
    #         err  = mean_squared_error(avg_hist, real_freq)
    #         if time == 0:
    #             first_error = err
    #         if time > 0 and err < errors[-1]:
    #             note = 'GOOD'
    #         else:
    #             note = 'BAD'
    #         errors.append(err)
    #         # if time > 0 and err < first_error:
    #         #     break
            
            
    #         print(f'At time {time}: {avg_hist}, error:{err}, note: {note}')

    #     if errors[0] > errors[-1]:
    #         accuracy += 1
    #         print(f'Last estimation is  better than the first')
    #     else:
    #         print(f'First estimation is better than the last')
        
    #     plt.plot(errors)
    #     plt.savefig(f'errors_nusers_{n}_exp_instance_{exp_ins}_accuracy_{accuracy}.png')

    # print(accuracy)
        









    num_seed = 10
    eps_perm = 2
    eps_min = 0.1
    eps_maj = 0.4
    n = 100000
    k=2
    data = gen_binary_data(n=n, one_prob=0.3)
    dic_mse = {seed: {
        "LGRRmaj1": [],
        "LGRRmin": [],
        "LGRRmaj2": [],
    } for seed in range(num_seed)}

    real_freq = np.unique(data, return_counts=True)[-1] / n
    print(f'Real freq {real_freq}')
    for seed in range(num_seed):
        
        grr_reports = [L_GRR_Client(input_data, k, eps_perm, eps_min) for input_data in data['v']]
        grr_est_freq = L_GRR_Aggregator_IBU(grr_reports, k, eps_perm, eps_min)
        print(f'Estimated frequency at seed: {seed}, epsilon: {eps_min}, ==> {grr_est_freq}')
        dic_mse[seed]["LGRRmin"].append(mean_squared_error(real_freq, grr_est_freq))



    # -------------- ##


    for seed in range(num_seed):
        majority_users = []
        minority_users = []

        for u in data['v']:
            if u == 0:
                report = L_GRR_Client(u, k, eps_perm, eps_maj)
                majority_users.append(report)
            else:
                report = L_GRR_Client(u, k, eps_perm, eps_min)
                minority_users.append(report)
        
        grr_est_freq_maj = L_GRR_Aggregator_IBU(majority_users, k, eps_perm, eps_maj)
        grr_est_freq_min = L_GRR_Aggregator_IBU(minority_users, k, eps_perm, eps_min)
        n_maj = len(majority_users)
        n_min = len(minority_users)
        weighted = (n_maj * grr_est_freq_maj + grr_est_freq_min * n_min) / (n)
        print(f'Separate {grr_est_freq_maj}: {grr_est_freq_min}')
        print(f'Estimated frequency at seed: {seed}, epsilon: {eps_maj}, ==> {weighted}')
        dic_mse[seed]["LGRRmaj1"].append(mean_squared_error(real_freq, weighted))



    eps_maj = 1

    for seed in range(num_seed):
        majority_users = []
        minority_users = []

        for u in data['v']:
            if u == 0:
                report = L_GRR_Client(u, k, eps_perm, eps_maj)
                majority_users.append(report)
            else:
                report = L_GRR_Client(u, k, eps_perm, eps_min)
                minority_users.append(report)
        
        grr_est_freq_maj = L_GRR_Aggregator_IBU(majority_users, k, eps_perm, eps_maj)
        grr_est_freq_min = L_GRR_Aggregator_IBU(minority_users, k, eps_perm, eps_min)
        n_maj = len(majority_users)
        n_min = len(minority_users)
        weighted = (n_maj * grr_est_freq_maj + grr_est_freq_min * n_min) / (n)
        print(f'Separate {grr_est_freq_maj}: {grr_est_freq_min}')
        print(f'Estimated frequency at seed: {seed}, epsilon: {eps_maj}, ==> {weighted}')
        dic_mse[seed]["LGRRmaj2"].append(mean_squared_error(real_freq, weighted))


    mean_errorsmin = np.mean([dic_mse[seed]["LGRRmin"] for seed in range(num_seed)], axis=0)
    mean_errorsmaj1 = np.mean([dic_mse[seed]["LGRRmaj1"] for seed in range(num_seed)], axis=0)
    mean_errorsmaj2 = np.mean([dic_mse[seed]["LGRRmaj2"] for seed in range(num_seed)], axis=0)
    


    print(f'Mean grr errors: 1: {mean_errorsmin},  2:{mean_errorsmaj1}, 3:{mean_errorsmaj2}')



    # for seed in range(num_seed):
    #     for ep in eps:
    #         l_grr_reports = [L_GRR_Client(input_data, k, ep, 0.7 * ep) for input_data in data['v']]
    #         grr_est_freq = L_GRR_Aggregator_MI(l_grr_reports, k, 1.3 * ep, ep)
    #         print(f'Estimated frequency at seed: {seed}, epsilon: {ep}, ==> {grr_est_freq}')
    #         dic_mse[seed]["LGRR"].append(mean_squared_error(real_freq, grr_est_freq))

    # mean_errors = np.mean([dic_mse[seed]["LGRR"] for seed in range(num_seed)], axis=0)

    # print(f'Mean lgrr errors: {mean_errors}')



    '''
    REAL interaction between server and users
    '''
    # number_of_interactions = 3
    # num_seed = 1
    # dic_mse = {seed: {
    #     "LGRR": [],
    # } for seed in range(num_seed)}


    # epsilons_majority = [0.2, 0.5, 0.7]
    # epsilons_minority = [0.2, 0.25, 0.3]

    # for interaction in range(number_of_interactions):
    #     for seed in range(num_seed):
    #         if interaction == 0:
    #             eps_inf = 2.0
    #             l_grr_reports = np.array([L_GRR_Client(input_data, k, eps_inf, epsilons_majority[interaction]) for input_data in data['v']])
    #             permanent_reports = l_grr_reports[:,0]
    #             randomized_reports = l_grr_reports[:,1]
    #             ps = l_grr_reports[:, 2, 4]
    #             qs = l_grr_reports[:, 3, 5]
    #             grr_est_freq = L_GRR_Aggregator_MI(randomized_reports, k, eps_inf, epsilons_majority[interaction])
    #             print(f'Estimated frequency, epsilon perm: {eps_inf}, eps1: {epsilons_majority[interaction]} ==> {grr_est_freq}')
    #             dic_mse[seed]["LGRR"].append(mean_squared_error(real_freq, grr_est_freq))
    #             mean_errors = np.mean([dic_mse[seed]["LGRR"] for seed in range(num_seed)], axis=0)
    #             print(f'Mean lgrr errors: {mean_errors}')
    #             randomized_reports = randomized_reports.reshape(-1,1)
    #             # ps = ps.reshape(-1, 1)
    #             # qs = qs.reshape(-1, 1)
            
    #         else:

    #             rand_op = []
    #             p_list = []
    #             q_list = []
    #             majority = grr_est_freq.argmax()
    #             minority = grr_est_freq.argmin()
    #             for input in permanent_reports:
                    
    #                 if input == majority:
    #                     op, p, q  = L_GRR_Client_Interaction(input, k, eps_inf, epsilons_majority[interaction-1])
    #                 else:
    #                     op, p, q = L_GRR_Client_Interaction(input, k, eps_inf, epsilons_minority[interaction-1])
    #                 rand_op.append(op)
    #                 p_list.append(p)
    #                 q_list.append(q)
                
    #             randomized_reports = np.concatenate((randomized_reports, rand_op.reshape(-1,1)), axis=1)
    #             ps = np.concatenate((ps, p.reshape(-1, 1)), axis=1)
    #             qs = np.concatenate((qs, q.reshape(-1, 1)), axis=1)

    #             grr_est_freq = GRR_Aggregator_IBU_interact(randomized_reports, k, grr_est_freq, epsilons_minority, epsilons_majority, interaction, ps, qs)


                    



                






