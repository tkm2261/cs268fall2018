import os
import pickle
import numpy as np
from utils import score_path
from sklearn.neighbors import NearestNeighbors

from logging import getLogger

logger = getLogger(__name__)


class Greedy:

    def __init__(self, n_cities, city_coods, filepath):
        self.n_cities = n_cities
        self.filepath = filepath
        self.city_coords = city_coods
        logger.info('start NN fit')
        self.nn = NearestNeighbors(n_neighbors=n_cities)
        logger.info('end NN fit')
        self.nn.fit(city_coods)

    def solve(self):
        logger.debug('enter')
        is_visited = np.zeros(self.n_cities, dtype=np.bool)
        is_visited[0] = True
        current_city = 0
        route = [current_city]
        start_k = 1
        while 1:
            for i in range(start_k, 7):
                k = 10**i
                if k > self.n_cities:
                    k = self.n_cities
                points = self.nn.kneighbors([self.city_coords[current_city]], n_neighbors=k, return_distance=False)[0]
                points = points[~is_visited[points]]
                if len(points) > 0:
                    break
            start_k = i
            if points.shape[0] == 0:
                break
            current_city = points[0]

            is_visited[current_city] = True
            route.append(current_city)
            if len(route) % 1000 == 0:
                logger.info(f'progress: {len(route)}')
        route += [0]
        self.route = route
        logger.debug(f'route: {route}')
        score = score_path(self.filepath, route)
        logger.info(f'{self.filepath} score: {len(route)} {score}')
        pd.Series(route, name='Path').to_csv(self.filepath + '.greedy.sub.csv', index=False)
        logger.debug('exit')
        return route


if __name__ == '__main__':
    import pandas as pd
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
    gd = Greedy(df.shape[0], df[['X', 'Y']].values, filepath)
    gd.solve()

    filepath = 'cities10000.csv'
    df = pd.read_csv(filepath)
    gd = Greedy(df.shape[0], df[['X', 'Y']].values, filepath)
    gd.solve()

    filepath = 'cities.csv'
    df = pd.read_csv(filepath)
    gd = Greedy(df.shape[0], df[['X', 'Y']].values, filepath)
    gd.solve()
