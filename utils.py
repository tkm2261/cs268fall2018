import pandas as pd
import numpy as np
from sympy import isprime, primerange

pnums = [i for i in primerange(0, 197770)]
# https://www.kaggle.com/thexyzt/xyzt-s-visualizations-and-various-tsp-solvers
cities = pd.read_csv('cities.csv', index_col=['CityId'])


def score_path(filepath, path):
    cities = pd.read_csv(filepath, index_col=['CityId'])

    path_df = cities.reindex(path).reset_index()

    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 +
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step +
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()


def score_path_santa(path):
    path_df = cities.reindex(path).reset_index()

    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 +
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step +
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()
