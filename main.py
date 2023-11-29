import pandas as pd
import numpy as np
from data import gen_binary_data
from protocols.grr import GRR_Client, GRR_Aggregator_MI, GRR_Aggregator_IBU
from protocols.L_GRR import L_GRR_Client, L_GRR_Aggregator_MI, L_GRR_Client_Interaction
from sklearn.metrics import mean_squared_error
from user import User
from server import Server
import matplotlib.pyplot as plt
import random

if __name__=='__main__':
    
    number_of_users = [100, 1000, 10000]
    number_of_users = [100000]
    for n in number_of_users:
        accuracy = 0
         
        plt.clf()
        # evolving_histogram_exp = []
        for exp_ins in range(1):
            data = gen_binary_data(n=n, one_prob=0.3)
            k = 2
            real_freq = np.unique(data, return_counts=True)[-1] / n
            num_seed = 5

            
            print(f'Real frequency {real_freq}')
            epsilons = [[2,0.1], [0.1, 0.6], [0.1,0.8],[0.1,1], [0.1,1], [0.1,1],[0.1,1],[0.1,1],[0.1,1.2],[0.1,1.4],[0.1,1.4],[0.1,1.4]]
            

            epsilons = [[2.6,0.5],[0.5,0.8],[0.5,1],[0.5,1.5],[0.5,2], [0.5,2.5],[0.5,2.5],[1, 2.5], [1.5,2.5]] + 200 * [[2,2.5]]
            # epsilons = [[2.6,0.5],[0.5,0.8],[0.5,1],[0.5,1.5]]

            times = len(epsilons)
            
            estimated_histogram = None
            average_histograms = []
            evolving_histogram_seed = []    
            dic_mse = {seed: {
                
            } for seed in range(num_seed)}

            for seed in range(num_seed):

                
            #permanent randomization phase
                server = Server(epsilons, k)
                eps, p1, q1 = server.get_permanent_randomization_configs()
                
                evolving_histogram = []
                #instantiate all users with their private value
                users = [User(v) for v in data['v']]

                #permanent randomization with high epsilon in the beginning
                permanent_values = [user.permanent_randomization(p1, q1) for user in users]

                #number of interactions between user and server
                for time in range(times):
                    #everyone uses same epsilon or privacy mechanism for the first time
                    if time == 0:
                        eps_perm, p1, q1, eps_1, p2, q2 = server.get_instantaneous_randomization_configs(time)
                        reports = []
                        random_users_index = random.sample(list(range(n)), int(n*0.2))
                        for u_ind, user in enumerate(users):
                            if True:#u_ind in random_users_index:
                                report = user.instantaneous_randomization(p1, q1, p2, q2)
                                reports.append(report)
                        estimated_histogram = server.get_estimated_histogram(reports, time)
                        print(time)
                        dic_mse[seed][0] = mean_squared_error(real_freq, estimated_histogram)

                    else:

                        #at time > 0, there will be a latest histogram that gives information to user about maj and min
                        eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj = server.get_instantaneous_randomization_configs(time)
                        majority = estimated_histogram.argmax()
                        minority = estimated_histogram.argmin()
                        reports = []
                        for u in users:
                            # if user's private value belongs to majority they'll apply p_maj and q_maj corresponding to higher epsilon
                            if u.private_value == majority:
                                report = u.instantaneous_randomization(p1, q1, p_maj, q_maj)
                            else:
                                report = u.instantaneous_randomization(p1, q1, p_min, q_min)
                            reports.append(report)
                        estimated_histogram = server.get_estimated_histogram(reports, time, private_list = np.array(data['v']))
                        print(f'Seed: {seed}, time: {time}, estimated: {estimated_histogram}')
                        dic_mse[seed][time] = mean_squared_error(real_freq, estimated_histogram)

                    evolving_histogram.append(estimated_histogram)
                evolving_histogram_seed.append(evolving_histogram)
            evolving_histogram_seed = np.array(evolving_histogram_seed)

            # evolving_histogram_exp.append(evolving_histogram)
        # evolving_histogram_exp = np.array(evolving_histogram_exp)


            errors = []
            first_error = None
            for time in range(times):
                errors.append(np.mean([dic_mse[seed][time] for seed in range(num_seed)]))
            print(errors)
            plt.plot(errors)
            plt.savefig(f'errors_nusers_{n}_exp_instance_{exp_ins}_accuracy_{accuracy}.png')

        print(accuracy)
            

























    # for seed in range(num_seed):
    #     for ep in eps:
    #         grr_reports = [GRR_Client(input_data, k, ep) for input_data in data['v']]
    #         grr_est_freq = GRR_Aggregator_IBU(grr_reports, k, ep)
    #         print(f'Estimated frequency at seed: {seed}, epsilon: {ep}, ==> {grr_est_freq}')
    #         dic_mse[seed]["GRR"].append(mean_squared_error(real_freq, grr_est_freq))

    # mean_errors = np.mean([dic_mse[seed]["GRR"] for seed in range(num_seed)], axis=0)

    # print(f'Mean grr errors: {mean_errors}')

    # dic_mse = {seed: {
    #     "LGRR": [],
    # } for seed in range(num_seed)}

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


                    



                






