import pandas as pd
import numpy as np
from data import gen_binary_data
from protocols.grr import GRR_Client, GRR_Aggregator_MI, GRR_Aggregator_IBU
from protocols.L_GRR import L_GRR_Client, L_GRR_Aggregator_MI, L_GRR_Client_Interaction
from sklearn.metrics import mean_squared_error
from user import User
from server import Server
import matplotlib.pyplot as plt
from utils import plot_errors
import random
import sys
import logging
textfile = 'log'
logging.basicConfig(filename=textfile, level=logging.INFO)
if __name__=='__main__':
    
    number_of_users = [10000]
    probs = [0.1,0.2,0.3,0.4,0.49]
    # probs = [0.4, 0.49]
    
    # all_times = [2,4,6,8,10]
    all_times = [22,24,26,28,30]

    e0s = [1,0.8,0.6,0.4]
    experiment_types = ['time', 'single', 'baseline']
    num_seed = 30

    '''
    Testing
    '''
    e0s = [1,0.8,0.6]
    all_times = [2,6,10,14]
    probs = [0.1, 0.3, 0.49]
    experiment_types = ['single', 'baseline','time']
    num_seed = 30
    '''
    END OF TESTING
    '''


    # experiment_types = ['baseline']
    for n in number_of_users:
        for e0 in e0s:
            for times in all_times:
                all_errors = []
                ldp_errors = []
                baseline_errors = []
                for minority_prob in probs:
                    accuracy = 0
                    for exp_ins in experiment_types:
                        plt.clf()
                        data = gen_binary_data(n=n, one_prob=minority_prob)
                        k = 2
                        real_freq = np.unique(data, return_counts=True)[-1] / n
                        
                        print(f'Real frequency {real_freq}')
                        eps_inf = 2.6
                        estimated_histogram = None
                        total = 1.243746262157857
                        if exp_ins == 'time':
                            e1 = ( np.log((1-np.exp(total+eps_inf))/(np.exp(total)-np.exp(eps_inf))) - e0) * 1/(times-1)
                            epsilons = np.array([[eps_inf, e0]] + [[0.8, e1] for _ in range(times-1)])
                        
                        if exp_ins == 'single': 
                            # e1 = 1.52
                            epsilons = [[eps_inf,1.52]]

                        if exp_ins == 'baseline':
                            epsilons = [[eps_inf, eps_inf, 1.52]]
                            estimated_histogram = real_freq[:]
                        
                        
                        average_histograms = []
                        dic_mse = {seed: {
                            } for seed in range(num_seed)
                        }
                        # np.random.seed(100)
                        for seed in range(num_seed):
                            #permanent randomization phase
                            server = Server(epsilons, k)
                            eps_perm, p1, q1 = server.get_permanent_randomization_configs()
                            
                            evolving_histogram = []
                            #instantiate all users with their private value
                            users = [User(v) for v in data['v']]
                            total_maj_loss = total_min_loss = 0
                            #permanent randomization with high epsilon in the beginning
                            permanent_values = [user.permanent_randomization(eps_perm) for user in users]
                            
                            #number of interactions between user and server
                            for time in range(len(epsilons)):
                                #everyone uses same epsilon or privacy mechanism for the first time
                                if exp_ins == 'baseline':
                                    majority = estimated_histogram.argmax()
                                    minority = estimated_histogram.argmin()
                                    reports = []
                                    p1, q1, p_maj, q_maj, p_min, q_min, eps_inf, eps_maj, eps_min = server.get_baseline_configurations()

                                    for u in users:
                                        if u.private_value == majority:
                                            report = u.instantaneous_randomization(eps_maj)
                                        else:
                                            report = u.instantaneous_randomization(eps_min)
                                        reports.append(report)
                                    estimated_histogram = server.get_baseline_estimated_histogram(reports, real_freq)
                                    if estimated_histogram[0] < estimated_histogram[1]:
                                        print('flipped')
                                    dic_mse[seed][time] = mean_squared_error(real_freq, estimated_histogram)

                                elif time == 0:
                                    eps_perm, p1, q1, eps_1, p2, q2 = server.get_instantaneous_randomization_configs(time)
                                    reports = []
                                    for u_ind, user in enumerate(users):
                                        report = user.instantaneous_randomization(eps_1)
                                        reports.append(report)
                                    estimated_histogram = server.get_estimated_histogram(reports, time, data['v'])
                                    if estimated_histogram[0] < estimated_histogram[1]:
                                        print('flipped')
                                    dic_mse[seed][0] = mean_squared_error(real_freq, estimated_histogram)
                                else :
                                    #at time > 0, there will be a latest histogram that gives information to user about maj and min
                                    eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj = server.get_instantaneous_randomization_configs(time)
                                    majority = estimated_histogram.argmax()
                                    minority = estimated_histogram.argmin()
                                    # majority_frequency = estimated_histogram[majority]
                                    # minority_frequency = estimated_histogram[minority]
                                    reports = []
                                    for u in users:
                                        if u.private_value == majority:
                                            # if np.random.binomial(1, majority_frequency):
                                            #     report = u.instantaneous_randomization(p1, q1, p_maj, q_maj)
                                            # else:
                                            #     report = u.instantaneous_randomization(p1, q1, p_min, q_min)
                                            report = u.instantaneous_randomization(eps_maj)
                                        else:
                                            # if np.random.binomial(1, minority_frequency):
                                            #     report = u.instantaneous_randomization(p1, q1, p_maj, q_maj)
                                            # else:
                                            #     report = u.instantaneous_randomization(p1, q1, p_min, q_min)
                                            report = u.instantaneous_randomization(eps_min)
                                        reports.append(report)
                                    estimated_histogram = server.get_estimated_histogram(reports, time, data['v'])

                                    if estimated_histogram[0] < estimated_histogram[1]:
                                        print('flipped')
                                    dic_mse[seed][time] = mean_squared_error(real_freq, estimated_histogram)
                                print(f'Seed: {seed}, Real freq: {real_freq}, users: {n}, time: {time}, estimated: {estimated_histogram}')

                        errors = []
                        first_error = None
                        for time in range(len(epsilons)):
                            errors.append(np.mean([dic_mse[seed][time] for seed in range(num_seed)]))
                        x_axis = [int(i+1) for i in range(len(epsilons))]
                        if exp_ins == 'time':
                            all_errors.append(errors)
                        elif exp_ins == 'single':
                            ldp_errors.append(errors[0])
                        else:
                            baseline_errors.append(errors[0])
                            
                        # all_errors.append(errors) if exp_ins == 'time' else ldp_errors.append(errors[0])
                    print(probs)
                # logging.info(f'epsilon0: {e0}, epsilon1: {e1}, interactions: {times}')
                logging.info(all_errors)
                logging.info(ldp_errors)
                logging.info(baseline_errors)
                plot_errors(probs, all_errors,  ldp_errors, baseline_errors, e0, e1, times, n)