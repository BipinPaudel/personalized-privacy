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

    def get_p1_q1(self):
        eps_perm = self.epsilons[0][0]
        p1 = np.exp(eps_perm) / (np.exp(eps_perm) + self.k - 1)
        q1 = (1 - p1) / (self.k - 1)
        return eps_perm, p1, q1


    def get_permanent_randomization_configs(self):
        return self.get_p1_q1()
    
    def get_instantaneous_randomization_configs(self, t):
        eps_perm, p1, q1 = self.get_p1_q1()
        if t == 0:
        # GRR parameters for round 2
            eps_1 = self.epsilons[t][1]
            #calculating probability for instantaneous randomization from epsilon_1 for time = 0
            p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + self.k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(self.k-1)+q1)
            q2 = (1 - p2) / (self.k-1)
            return eps_perm, p1, q1, eps_1, p2, q2
        
        #calculating probability for instantaneous randomization from epsilon_maj and epsilon_min for time > 0
        eps_min = self.epsilons[t][0]
        eps_maj = self.epsilons[t][1]
        p_maj = (q1 - np.exp(eps_maj) * p1) / ((-p1 * np.exp(eps_maj)) + self.k*q1*np.exp(eps_maj) - q1*np.exp(eps_maj) - p1*(self.k-1)+q1)
        q_maj = (1 - p_maj) / (self.k-1)

        p_min = (q1 - np.exp(eps_min) * p1) / ((-p1 * np.exp(eps_min)) + self.k*q1*np.exp(eps_min) - q1*np.exp(eps_min) - p1*(self.k-1)+q1)
        q_min = (1 - p_min) / (self.k-1)

        return eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj

    def estimate_first_time(self, nr):
        eps_perm = self.epsilons[0][0]
        eps_1 = self.epsilons[0][1]

        return L_GRR_Aggregator_MI(nr, self.k, eps_perm, eps_1)

    '''
    size(self.noisy_reports_probability) = (times,n,2), at each time, for each user, calculates the prob that 
    user belongs to majority and minority


    '''
    def get_estimated_histogram(self, nr, time, private_list=None):
        if time == 0:
            estimated_histogram = self.perform_first_time_estimation(nr, time)
            return estimated_histogram
        else:
            # print(f'time: {time}')
            estimated_histogram = self.perform_other_time_estimation(nr, time, private_list)
            return estimated_histogram
        

    
    def perform_first_time_estimation(self, nr, time):
        # stores user submitted information for all times
        self.noisy_reports = np.array(nr, dtype=int)
        
        hist = self.estimate_first_time(nr)
        # stores estimated histogram for all times
        self.histograms = np.array([hist])

        #calculating P(B^1|B=0) and P(B^1|B=1) for all users
        probability_at_t = []
        for i in range(len(nr)):
            latestnr = nr[i] #latestnr corresponds to latest submitted value by the user
            eps_perm, p1, q1, eps_1, p2, q2 = self.get_instantaneous_randomization_configs(time)
            both_prob = []
            for j in range(self.k):
                if latestnr == j:
                    both_prob.append(p1 * p2 + q1 * q2)
                else:
                    both_prob.append(p1 * q2 + p2 * q1)
            probability_at_t.append(both_prob)

        # stores P(B^T|B=0) and P(B^T|B=1) for all users at all times
        self.noisy_reports_probability = np.array([probability_at_t])
        return hist

    def perform_other_time_estimation(self, nr, time, private_list=None):
            
        eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj = self.get_instantaneous_randomization_configs(time)
        self.noisy_reports = np.vstack((self.noisy_reports, nr))

        

        # latest_hist = self.histograms[-1]

        # majority = latest_hist.argmax()
        # minority = latest_hist.argmin()
        
        # #for majority users : latest exp
        # n_maj = int(latest_hist[majority] * len(nr))
        # count_report = np.zeros(self.k)
        # count_report[majority] = int((p1*p_maj + q1 * q_maj) * n_maj)
        # count_report[minority] = int((p1*q_maj + q1 * p_maj) * n_maj)
        # majority_freq = MI_long(count_report, n_maj, p1, q1, p_maj, q_maj)
        # #for minority users
        # n_min = int(latest_hist[minority] * len(nr))
        # count_report = np.zeros(self.k)
        # count_report[majority] = int((p1*p_min + q1 * q_min) * n_min)
        # count_report[minority] = int((p1*q_min + q1 * p_min) * n_min)
        # minority_freq = MI_long(count_report, n_min, p1, q1, p_min, q_min)

        
        # weighted_frequency = (majority_freq * n_maj  + minority_freq * n_min) / (n_maj + n_min)
        # self.histograms = np.vstack((self.histograms, weighted_frequency))
        # return self.histograms[-1]
        # majority_noisy_reports,minority_noisy_reports  = self.classify_users(eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj)
        

        freq = self.classify_users_and_provide_freq(eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj, private_list)
        # # h = np.unique(arrs, return_counts=True)[-1] / len(self.noisy_reports)
        self.histograms = np.vstack((self.histograms, freq))
        return self.histograms[-1]
        
        '''with perfect knowledge'''
        # nr = np.array(nr)
        # majority_noisy_reports = nr[private_list==self.histograms[-1].argmax()]
        # minority_noisy_reports = nr[private_list==self.histograms[-1].argmin()]
        
        # majority_freq = L_GRR_Aggregator_IBU(majority_noisy_reports, self.k, eps_perm, eps_maj)
        # minority_freq = np.array([0, 0])
        
        # if len(minority_noisy_reports) > 0:
        # minority_freq = L_GRR_Aggregator_IBU(minority_noisy_reports, self.k, eps_perm, eps_min)
        # print(f'Buckets: {len(majority_noisy_reports)}:{len(minority_noisy_reports)}')
        # weighted_frequency = (majority_freq * len(majority_noisy_reports) + minority_freq * len(minority_noisy_reports)) / (len(majority_noisy_reports) + len(minority_noisy_reports))
        # self.histograms = np.vstack((self.histograms, weighted_frequency))
        # return self.histograms[-1]

   
    def classify_users_and_provide_freq(self, eps_perm, p1, q1, eps_min, p_min, q_min, eps_maj, p_maj, q_maj, private_list=None):
        """
        Goal is to classify whether user belongs to majority or minority based on the submitted value at all times
        """


        prob_of_users_in_majority = self.calc_user_prob_majority(p1, q1, p_min, q_min, p_maj, q_maj, private_list)
        
        binomial_array = np.random.binomial(1, prob_of_users_in_majority)
        return np.unique(1-binomial_array, return_counts=True)[-1] / len(binomial_array)
        # return np.array([len(binomial_array[binomial_array==1]), len(binomial_array[binomial_array==0])])/len(self.noisy_reports[-1])

        # majority_bin = latest_nr_report[binomial_array==1]
        # minority_bin = latest_nr_report[binomial_array==0]

        return majority_bin, minority_bin

    
    def calc_user_prob_majority(self,p1, q1, p_min, q_min, p_maj, q_maj, private_list=None):
        """
        Calculates the probability whether the user belongs to majority or minority using bayesian technique 

        outputs: P(B=majority|B^1, B^2, ....)
        """


        latest_histogram = self.histograms[-1]
        
        latest_nr_report = self.noisy_reports[-1]

        majority = latest_histogram.argmax()
        minority = latest_histogram.argmin()
        probability_at_t = []
        
        
        #for each user report calculate P(B^t|B=0) and P(B^t|B=1)
        for n in range(len(latest_nr_report)):
            both_prob = []
            latestnr = latest_nr_report[n]

            #for majority and minority separately, calculates P(B^t|B) for t = 1...
            for b in range(self.k):

                if b == majority:
                    if latestnr == b:
                        probb = p1 * p_maj + q1 * q_maj
                    else:
                        probb = p1 * q_maj + q1 * p_maj
                else:
                    if latestnr == b:
                        probb = p1 * p_min + q1 * q_min
                    else:
                        probb = p1 * q_min + q1 * p_min
                both_prob.append(probb)
            probability_at_t.append(both_prob)
        

        self.noisy_reports_probability = np.vstack((self.noisy_reports_probability, np.array([probability_at_t])))
        prob_of_users_in_majority = []

        """
        For each user submitted value, calculate P(B=majority| B^1, B^2.....) using Bayesian technique
        """
        for n in range(len(latest_nr_report)):
            latestnr = latest_nr_report[n]
            user_n_probs = self.noisy_reports_probability[:,n]

            # numerator = P(B=majority) * P(B^1|B=majority) * P(B^2|B=majority)

            numerator = latest_histogram[majority] * np.prod(user_n_probs[:,majority])

            denominator = 0

            # denominator = sum over B = (majority and minority), P(P(B) * P(B^1|B) * P(B^2|B))
            for b in range(self.k):
                denominator += latest_histogram[b] * np.prod(user_n_probs[:,b])

            p_user_majority = numerator/denominator
            prob_of_users_in_majority.append(p_user_majority)
        return np.array(prob_of_users_in_majority)






        
        


    
