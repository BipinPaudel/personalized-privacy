import numpy as np
from protocols.grr import GRR_Client, GRR_Aggregator_MI, GRR_Aggregator_IBU
from sklearn.metrics import mean_squared_error
from protocols.L_GRR import L_GRR_Client, L_GRR_Aggregator_MI, L_GRR_Client_Interaction

class User:

    def __init__(self, private_value) -> None:
        self.private_value = private_value


    def permanent_randomization(self, p1, q1):
        eps_perm = np.log(p1/q1)
        k = 2
        self.permanent_value = GRR_Client(self.private_value, k, eps_perm)
        return self.permanent_value
    
    def instantaneous_randomization(self, p1, q1, p2, q2):
        eps_inst = np.log(p2/q2)
        k = 2
        return GRR_Client(self.permanent_value, k, eps_inst)
    
    

    