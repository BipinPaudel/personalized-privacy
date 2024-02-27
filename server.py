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

    @staticmethod
    def calc_p_q_from_epsilon(epsilon, k):
        p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        q = (1 - p) / (k - 1)
        return p, q
    
    @staticmethod
    def calc_maj_min_probability(eps_maj, eps_min, k):
        p_maj, q_maj = Server.calc_p_q_from_epsilon(eps_maj, k)
        p_min, q_min = Server.calc_p_q_from_epsilon(eps_min, k)
        
        # (q1 - np.exp(eps_maj) * p1) / ((-p1 * np.exp(eps_maj)) + k*q1*np.exp(eps_maj) - q1*np.exp(eps_maj) - p1*(k-1)+q1)
        # q_maj = (1 - p_maj) / (k-1)

        # p_min = (q1 - np.exp(eps_min) * p1) / ((-p1 * np.exp(eps_min)) + k*q1*np.exp(eps_min) - q1*np.exp(eps_min) - p1*(k-1)+q1)
        # q_min = (1 - p_min) / (k-1)

        return p_maj, q_maj, p_min, q_min

    
    def get_p1_q1(self):
        eps_perm = self.epsilons[0][0]
        p1, q1 = Server.calc_p_q_from_epsilon(eps_perm, self.k)        
        return eps_perm, p1, q1


    def get_permanent_randomization_configs(self):
        return self.get_p1_q1()
    
    def get_instantaneous_randomization_configs(self, t):
        eps_perm, p1, q1 = self.get_p1_q1()
        if t == 0:
        # GRR parameters for round 2
            eps_1 = self.epsilons[t][1]
            #calculating probability for instantaneous randomization from epsilon_1 for time = 0
            # p3 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + self.k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(self.k-1)+q1)
            # q3 = (1 - p3) / (self.k-1)
            p2, q2 = Server.calc_p_q_from_epsilon(eps_1, self.k)
            return eps_perm, p1, q1, eps_1, p2, q2
        
        #calculating probability for instantaneous randomization from epsilon_maj and epsilon_min for time > 0
        eps_min = self.epsilons[t][0]
        eps_maj = self.epsilons[t][1]

        p_maj, q_maj, p_min, q_min = Server.calc_maj_min_probability(eps_maj, eps_min, self.k)

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
    
    def get_baseline_configurations(self, ):
        eps_inf = self.epsilons[0][0]
        eps_maj = self.epsilons[0][1]
        eps_min = self.epsilons[0][2]

        p1, q1 = Server.calc_p_q_from_epsilon(eps_inf, self.k)
        p_maj, q_maj, p_min, q_min = Server.calc_maj_min_probability(eps_maj, eps_min, self.k)
        return p1, q1, p_maj, q_maj, p_min, q_min, eps_inf, eps_maj, eps_min

    def get_baseline_estimated_histogram(self, nr, real_freq):
        self.noisy_reports = np.array([nr], dtype=int)
        self.prior = real_freq
        p1, q1, p_maj, q_maj, p_min, q_min, eps_inf, eps_maj, eps_min = self.get_baseline_configurations()
        p_b1_bstar = Server.calc_p_b_bstar(real_freq, p1, q1, p_maj, q_maj, p_min, q_min)
        self.p_b_bstar = np.array([p_b1_bstar])

        posterior = Server.calculate_user_posterior(nr, self.noisy_reports, real_freq, self.p_bstar_b, self.p_b_bstar)

        estimated = np.array([np.sum(posterior), len(nr)-np.sum(posterior)])
        estimated /= len(nr)
        return estimated



        
    def perform_first_time_estimation(self, nr, time, private_list=None):
        # stores user submitted information for all times
        self.noisy_reports = np.array([nr], dtype=int)
        self.prior = self.estimate_first_time(nr)
        # self.prior = np.unique(private_list, return_counts=True)[-1] / len(nr)
        # stores estimated histogram for all times
        # self.histograms = np.array([hist])
        # print(f'First time estimated: {self.prior}')
        # calculate p_b1_bstar
        eps_perm, p1, q1, eps_1, p2, q2 = self.get_instantaneous_randomization_configs(time)
        p_b1_bstar = np.array([[p2, q2],
                               [q2, p2]])
        self.p_b_bstar = np.array([p_b1_bstar])

        #calculating P(B=0|B_1)
        
        return self.prior
        # posterior = []
        # for user_index in range(len(nr)):
        #     user_disclosed = self.noisy_reports[:,user_index]
        #     numerator = self.prior[0]*self.p_bstar_b[0][0]* np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,0])  + self.prior[0]*self.p_bstar_b[1][0] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,1])
        #     d_b_0 = numerator
        #     d_b_1 = self.prior[1] * self.p_bstar_b[0][1] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,0]) + self.prior[1] * self.p_bstar_b[1][1] * np.prod(self.p_b_bstar[np.arange(len(self.p_b_bstar)),user_disclosed][:,1])
        #     denominator = d_b_0 + d_b_1
        #     posterior.append(numerator/denominator)
        
        # estimated = np.array([np.sum(posterior), len(nr)-np.sum(posterior)])
        # estimated /= len(nr)
        # return estimated
    
    def estimate_first_time(self, nr):
        eps_perm = self.epsilons[0][0]
        eps_1 = self.epsilons[0][1]
        return L_GRR_Aggregator_MI(nr, self.k, eps_perm, eps_1)
    
    @staticmethod
    def calc_p_b_bstar(prior, p1, q1, p_maj, q_maj, p_min, q_min):
        all_b = [0, 1]
        all_b_star = [0, 1]
        p_b_bstar = np.ones((2,2))
        for b in all_b:
            for b_star in all_b_star:
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
        return p_b_bstar
    
    @staticmethod
    def calculate_user_posterior(nr, noisy_reports, prior, p_bstar_b, p_b_bstar):
        posterior = []
        for user_index in range(len(nr)):
            user_disclosed = noisy_reports[:,user_index]
            numerator = prior[0]*p_bstar_b[0][0]* np.prod(p_b_bstar[np.arange(len(p_b_bstar)),user_disclosed][:,0])  + prior[0]*p_bstar_b[1][0] * np.prod(p_b_bstar[np.arange(len(p_b_bstar)),user_disclosed][:,1])
            d_b_0 = numerator
            d_b_1 = prior[1] * p_bstar_b[0][1] * np.prod(p_b_bstar[np.arange(len(p_b_bstar)),user_disclosed][:,0]) + prior[1] * p_bstar_b[1][1] * np.prod(p_b_bstar[np.arange(len(p_b_bstar)),user_disclosed][:,1])
            denominator = d_b_0 + d_b_1
            posterior.append(numerator/denominator)

        return posterior


    def perform_other_time_estimation(self, nr, time, private_list=None):
            
        eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj = self.get_instantaneous_randomization_configs(time)
        
        self.noisy_reports = np.vstack((self.noisy_reports, nr))
        
        # prior = self.priord.histograms[-1]


        prior =  self.histograms[-1]
        # weight = np.array([0.7,0.3])
        # prior = np.dot(weight,np.array([prior, self.prior]))
        p_b_bstar = Server.calc_p_b_bstar(prior, p1, q1, p_maj, q_maj, p_min, q_min)
        self.p_b_bstar = np.vstack((self.p_b_bstar, [p_b_bstar]))

        posterior = Server.calculate_user_posterior(nr, self.noisy_reports, prior, self.p_bstar_b, self.p_b_bstar)

        estimated = np.array([np.sum(posterior), len(nr)-np.sum(posterior)])
        estimated /= len(nr)
        return estimated


        

        

        
   
        


    
