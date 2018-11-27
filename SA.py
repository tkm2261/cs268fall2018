import os
import pickle
import numpy as np
from utils import score_path, score_path_santa
from sklearn.neighbors import NearestNeighbors

from logging import getLogger

logger = getLogger(__name__)


def _prob_bolzman(e1, e2, t):
    if e2 < e1:
        return 1
    else:
        return np.exp(- (e2 - e1) / t)


class SA:
    def __init__(self, n_cities, city_coods, filepath, score_func, init_sol=None):
        self.n_cities = n_cities
        self.filepath = filepath
        self.city_coords = city_coods
        self.init_sol = init_sol
        self.score_func = score_func

    def solve(self):
        np.random.seed(0)
        logger.debug('enter')
        if self.init_sol is None:
            self.init_sol = np.arange(1, self.n_cities)
            np.random.shuffle(self.init_sol)
        else:
            self.init_sol = np.array(self.init_sol, dtype=np.int)
        current_sol = np.array(self.init_sol)
        current_score = self.score_func([0] + current_sol.tolist() + [0])
        curr_temp = 100
        ending_temp = 0.01

        for i in range(100000):
            if i % 100 == 0:
                logger.info(f'itr: {i}, temp: {curr_temp}, score: {current_score}')
            curr_temp = curr_temp - 0.01
            if curr_temp < ending_temp:
                break
            swaps = np.random.randint(0, self.init_sol.shape[0] - 1, 2)
            next_sol = current_sol.copy()
            next_sol[swaps] = current_sol[swaps[::-1]]
            next_score = self.score_func([0] + current_sol.tolist() + [0])

            if next_score < current_score:
                current_sol = next_sol
                current_score = next_score
            else:
                prb = _prob_bolzman(current_score, next_score, curr_temp)
                if prb > np.random.random():
                    current_sol = next_sol
                    current_score = next_score

        self.route = [0] + current_sol.tolist() + [0]
        logger.debug(f'route: {self.route}')
        score = current_score
        logger.info(f'{self.filepath} score: {len(self.route)} {score}')
        pd.Series(self.route, name='Path').to_csv(self.filepath + '.sa.sub.csv', index=False)
        logger.debug('exit')

        return self.route


if __name__ == '__main__':
    import pandas as pd
    from functools import partial
    from logging import StreamHandler, DEBUG, Formatter, FileHandler, NullHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('DEBUG')
    handler.setFormatter(log_fmt)
    logger.setLevel('DEBUG')
    logger.addHandler(handler)

    handler = FileHandler(os.path.basename(__file__) + '.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    filepath = 'cities1000.csv'
    df = pd.read_csv(filepath)
    func = partial(score_path, filepath)
    gd = SA(df.shape[0], df[['X', 'Y']].values, filepath, func, init_sol=None)
    gd.solve()

    filepath = 'cities10000.csv'
    df = pd.read_csv(filepath)
    func = partial(score_path, filepath)
    gd = SA(df.shape[0], df[['X', 'Y']].values, filepath, func, init_sol=None)
    gd.solve()

    filepath = 'cities.csv'
    df = pd.read_csv(filepath)
    gd = SA(df.shape[0], df[['X', 'Y']].values, filepath, score_path_santa, init_sol=None)
    gd.solve()
