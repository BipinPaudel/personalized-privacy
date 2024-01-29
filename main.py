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
import sys
if __name__=='__main__':
    
    number_of_users = [10000]
    for n in number_of_users:
        accuracy = 0
        
        for exp_ins in range(2):
            data = gen_binary_data(n=n, one_prob=0.3)
            k = 2
            plt.clf()
            real_freq = np.unique(data, return_counts=True)[-1] / n
            num_seed = 20

            
            print(f'Real frequency {real_freq}')
            # epsilons = [[5,0.5], [0.7, 1], [0.8,1.5],[1, 2]] + [[1.5,2.4] for _ in range(1)]

            # epsilons = [[2.6,0.4],[minority_epsilon,0.8],[minority_epsilon,1],[minority_epsilon,1.5],
            #              [minority_epsilon,1.8]] + [[minority_epsilon,2] for _ in range(25)]
            eps_inf = 2.6
            # e1 = 0.38
            # epsilons = np.array([[2.6, e1], [0.8, e1], [0.8, e1], [0.8, e1]])
            
            #### temp
            e1 = 0.5067
            epsilons = np.array([[2.6, e1]] + [[0.8, e1]]*3)


            

            if exp_ins == 1: 
                epsilons = [[2.6,1.52]]
            # epsilons = [[2.6,0.4],[0.4,0.8],[0.4,1],[0.4,1.5], [0.4,1.8], [0.4,2], [0.4,2], [0.4,2], [0.4,2]]

            # epsilons = [[2.6,0.01],[0.01,0.8],[0.01,1],[0.01,1.5],[0.01,2], [0.01,2.5],[0.01,2.5],[0.01, 2.5], [0.01,2.5]] + 20 * [[0.01,2.5]]
            # epsilons = [[2.6,0.5],[0.5,0.8],[0.5,1],[0.5,1.5]]

            times = len(epsilons)
            
            estimated_histogram = None
            average_histograms = []
            dic_mse = {seed: {
                } for seed in range(num_seed)
            }
            # np.random.seed(100)
            for seed in range(num_seed):

                
                #permanent randomization phase
                server = Server(epsilons, k)
                eps, p1, q1 = server.get_permanent_randomization_configs()
                
                evolving_histogram = []
                #instantiate all users with their private value
                users = [User(v) for v in data['v']]
                total_maj_loss = total_min_loss = 0
                #permanent randomization with high epsilon in the beginning
                permanent_values = [user.permanent_randomization(p1, q1) for user in users]
                
                #number of interactions between user and server
                for time in range(times):
                    #everyone uses same epsilon or privacy mechanism for the first time
                    if time == 0:
                        eps_perm, p1, q1, eps_1, p2, q2 = server.get_instantaneous_randomization_configs(time)
                        reports = []
                        for u_ind, user in enumerate(users):
                            report = user.instantaneous_randomization(p1, q1, p2, q2)
                            reports.append(report)
                        estimated_histogram = server.get_estimated_histogram(reports, time, data['v'])
                        if estimated_histogram[0] < estimated_histogram[1]:
                            print('flipped')
                            sys.exit()
                        dic_mse[seed][0] = mean_squared_error(real_freq, estimated_histogram)
                        
                    else:
                        #at time > 0, there will be a latest histogram that gives information to user about maj and min
                        eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj = server.get_instantaneous_randomization_configs(time)
                        majority = 0 #estimated_histogram.argmax()
                        minority = 1 #estimated_histogram.argmin()
                        reports = []
                        for u in users:
                            # if user's private value belongs to majority they'll apply p_maj and q_maj corresponding to higher epsilon
                            
                            # if u.private_value == majority:
                            #     report = u.instantaneous_randomization(p1, q1, p_maj, q_maj)
                            # else:
                            #     report = u.instantaneous_randomization(p1, q1, p_min, q_min)

                            if u.private_value == majority:
                                if np.random.binomial(1, estimated_histogram[0]):
                                    report = u.instantaneous_randomization(p1, q1, p_maj, q_maj)
                                else:
                                    report = u.instantaneous_randomization(p1, q1, p_min, q_min)
                            else:
                                if np.random.binomial(1, estimated_histogram[1]):
                                    report = u.instantaneous_randomization(p1, q1, p_maj, q_maj)
                                else:
                                    report = u.instantaneous_randomization(p1, q1, p_min, q_min)
                            reports.append(report)
                        estimated_histogram = server.get_estimated_histogram(reports, time, data['v'])
                        if estimated_histogram[0] < estimated_histogram[1]:
                            print('flipped')
                            sys.exit()
                        dic_mse[seed][time] = mean_squared_error(real_freq, estimated_histogram)
                    print(f'Seed: {seed}, Real freq: {real_freq}, users: {n}, time: {time}, estimated: {estimated_histogram}')




            errors = []
            first_error = None
            for time in range(times):
                errors.append(np.mean([dic_mse[seed][time] for seed in range(num_seed)]))
            for er in errors:
                print(f'{er:.8f}')
            x_axis = [int(i+1) for i in range(times)]
            plt.plot(x_axis, errors)
            first_txt = f'First error: {errors[0]:.8f}'
            last_txt = f'Last error: {errors[-1]:.8f}'
            plt.text(1,errors[0],first_txt+"\n"+last_txt)
            plt.savefig(f'errors_nusers_{n}_exp_instance_{exp_ins}_accuracy_{accuracy}.png')

        print(accuracy)