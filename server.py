import numpy as np
from protocols.L_GRR import L_GRR_Client, L_GRR_Aggregator_MI, L_GRR_Client_Interaction, L_GRR_Aggregator_IBU
from estimators.Histogram_estimator import MI_long, IBU
class Server:

    def __init__(self, epsilons, k) -> None:
        self.epsilons = epsilons
        self.k = k
        self.t = len(self.epsilons)
        self.noisy_reports = None
        self.histograms = None
        self.p_b_bstar = None
        self.hist = np.zeros(2)
        eps_perm, p1, q1 = self.get_p1_q1()
        self.p_bstar_b = np.array([
            [p1, q1], 
            [q1, p1]
        ])

    def calc_p1_q1(eps_perm, k):
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
        q1 = (1 - p1) / (k - 1)
        return p1, q1
    
    def get_p1_q1(self):
        eps_perm = self.epsilons[0][0]
        p1, q1 = Server.calc_p1_q1(eps_perm, self.k)        
        return eps_perm, p1, q1


    def get_permanent_randomization_configs(self):
        return self.get_p1_q1()
    
    def get_instantaneous_randomization_configs(self, t):
        eps_perm, p1, q1 = self.get_p1_q1()
        if t == 0:
        # GRR parameters for round 2
            eps_1 = self.epsilons[t][1]
            #calculating probability for instantaneous randomization from epsilon_1 for time = 0
            # p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + self.k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(self.k-1)+q1)
            # q2 = (1 - p2) / (self.k-1)
            p2 = np.exp(eps_1) / (np.exp(eps_1) + self.k - 1)
            q2 = (1 - p2) / (self.k - 1)
            return eps_perm, p1, q1, eps_1, p2, q2
        
        #calculating probability for instantaneous randomization from epsilon_maj and epsilon_min for time > 0
        eps_min = self.epsilons[t][0]
        eps_maj = self.epsilons[t][1]
        p_maj = (q1 - np.exp(eps_maj) * p1) / ((-p1 * np.exp(eps_maj)) + self.k*q1*np.exp(eps_maj) - q1*np.exp(eps_maj) - p1*(self.k-1)+q1)
        q_maj = (1 - p_maj) / (self.k-1)

        p_min = (q1 - np.exp(eps_min) * p1) / ((-p1 * np.exp(eps_min)) + self.k*q1*np.exp(eps_min) - q1*np.exp(eps_min) - p1*(self.k-1)+q1)
        q_min = (1 - p_min) / (self.k-1)

        return eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj

    '''
    size(self.noisy_reports_probability) = (times,n,2), at each time, for each user, calculates the prob that 
    user belongs to majority and minority
    '''
    def get_estimated_histogram(self, nr, time, private_list=None):
        if time == 0:
            estimated_histogram = self.perform_first_time_estimation(nr, time, private_list)
            self.histograms = np.array([estimated_histogram])
        else:
            estimated_histogram = self.perform_other_time_estimation(nr, time, private_list)
            self.histograms = np.vstack((self.histograms, estimated_histogram))
        
        return estimated_histogram



    def perform_first_time_estimation(self, nr, time, private_list=None):
        # stores user submitted information for all times
        self.noisy_reports = np.array([nr], dtype=int)
        self.prior = self.estimate_first_time(nr)
        # self.prior = np.unique(private_list, return_counts=True)[-1] / len(nr)
        # stores estimated histogram for all times
        # self.histograms = np.array([hist])
        print(f'First time estimated: {self.prior}')
        # calculate p_b1_bstar
        eps_perm, p1, q1, eps_1, p2, q2 = self.get_instantaneous_randomization_configs(time)
        p_b1_bstar = np.array([[p2, q2],
                               [q2, p2]])
        self.p_b_bstar = np.array([p_b1_bstar])

        #calculating P(B=0|B_1)
        p_b_b1 = []
        for user_index in range(len(nr)):
            user_disclosed = self.noisy_reports[:,user_index]
            numerator = self.prior[0]*self.p_bstar_b[0][0]* np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,0])  + self.prior[0]*self.p_bstar_b[1][0] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,1])
            d_b_0 = numerator
            d_b_1 = self.prior[1] * self.p_bstar_b[0][1] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,0]) + self.prior[1] * self.p_bstar_b[1][1] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,1])
            denominator = d_b_0 + d_b_1
            p_b_b1.append(numerator/denominator)
        
        estimated = np.array([np.sum(p_b_b1), len(nr)-np.sum(p_b_b1)])
        estimated /= len(nr)
        return estimated
    
    def estimate_first_time(self, nr):
        eps_perm = self.epsilons[0][0]
        eps_1 = self.epsilons[0][1]
        return L_GRR_Aggregator_MI(nr, self.k, eps_perm, eps_1)

    def perform_other_time_estimation(self, nr, time, private_list=None):
            
        eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj = self.get_instantaneous_randomization_configs(time)
        self.noisy_reports = np.vstack((self.noisy_reports, nr))
        p_b_bstar = np.ones((2,2))
        prior = self.prior#self.histograms[-1]
        
        all_b = [0, 1]
        all_b_star = [0, 1]

        for b in all_b:
            for b_star in all_b_star:
                # if b == 0 and b_star == 0:
                #     pp = p_maj * p1 * prior[0] + p_min * q1 * prior[1]

                # elif b == 1 and b_star == 0:
                #     pp = q_maj * p1 * prior[0] + q_min * q1 * prior[1]

                # elif b == 0 and b_star == 1:
                #     pp = q_maj * q1 * prior[0] + q_min * p1 * prior[1]

                # elif b == 1 and b_star == 1:
                #     pp = p_maj * q1 * prior[0] + p_min * p1 * prior[1]
                # if b == 0 and b_star == 0:
                #     pp = p_maj * prior[0] + p_min * prior[1]

                # elif b == 1 and b_star == 0:
                #     pp = q_maj *  prior[0] + q_min *  prior[1]

                # elif b == 0 and b_star == 1:
                #     pp = q_maj * prior[0] + q_min * prior[1]

                # elif b == 1 and b_star == 1:
                #     pp = p_maj * prior[0] + p_min * prior[1]

                if b_star == 0:
                    prob1 = (prior[0] * p1)/(prior[0] * p1 + prior[1]*q1)
                    prob2 = (prior[1] * q1)/(prior[1] * q1 + prior[0]*p1)
                    if b == 0:
                        pp = prob1 * p_maj + prob2 * p_min
                    else:
                        pp = prob1 * q_maj + prob2 * q_min

                else:
                    prob1 = (prior[0] * q1)/(prior[0] * q1 + prior[1]*p1)
                    prob2 = (prior[1] * p1)/(prior[0] * q1 + prior[1]*p1)
                    if b == 0:
                        pp = prob1 * q_maj + prob2 * q_min
                    else:
                        pp = prob1 * p_maj + prob2 * p_min

                p_b_bstar[b][b_star] = pp
        self.p_b_bstar = np.vstack((self.p_b_bstar, [p_b_bstar]))


        posterior = []
        for user_index in range(len(nr)):
            user_disclosed = self.noisy_reports[:,user_index]
            numerator = prior[0]*self.p_bstar_b[0][0]* np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,0])  + prior[0]*self.p_bstar_b[1][0] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,1])
            d_b_0 = numerator
            d_b_1 = prior[1] * self.p_bstar_b[0][1] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,0]) + prior[1] * self.p_bstar_b[1][1] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,1])
            denominator = d_b_0 + d_b_1
            posterior.append(numerator/denominator)


        estimated = np.array([np.sum(posterior), len(nr)-np.sum(posterior)])
        estimated /= len(nr)
        return estimated


        

        

        
   
        


    
