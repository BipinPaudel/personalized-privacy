import numpy as np

class User:

    def __init__(self, private_value) -> None:
        self.private_value = private_value


    def permanent_randomization(self, eps_perm):
        k = 2
        self.permanent_value = self.GRR_Client(self.private_value, k, eps_perm)
        return self.permanent_value
    
    def instantaneous_randomization(self, eps):
        k = 2
        return self.GRR_Client(self.permanent_value, k, eps)
    

    def GRR_Client(self, input_data, k, epsilon):
        """
        Generalized Randomized Response (GRR) protocol, a.k.a., direct encoding [1] or k-RR [2].

        :param input_data: user's true value;
        :param k: attribute's domain size;
        :param epsilon: privacy guarantee;
        :return: sanitized value.
        """

        # Validations
        if input_data < 0 or input_data >= k:
            raise ValueError('input_data (integer) should be in the range [0, k-1].')
        if not isinstance(k, int) or k < 2:
            raise ValueError('k needs an integer value >=2.')
        if epsilon > 0:
            
            # GRR parameters
            p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)

            # Mapping domain size k to the range [0, ..., k-1]
            domain = np.arange(k) 
            
            # GRR perturbation function
            if np.random.binomial(1, p) == 1:
                return input_data

            else:
                return np.random.choice(domain[domain != input_data])

        else:
            raise ValueError('epsilon needs a numerical value greater than 0.')
    
    

    