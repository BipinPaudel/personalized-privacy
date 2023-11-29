import numpy as np
import pandas as pd


def gen_binary_data(n=10000, one_prob=0.15):
        return pd.DataFrame(data = {'v':[np.random.binomial(1, one_prob) for _ in range(n)]})