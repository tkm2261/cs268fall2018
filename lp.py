import pulp
import pickle
from utils import score_path
from pulp import LpVariable, LpProblem, LpMinimize, COIN, lpSum, COINMP_DLL
from scipy.spatial import distance_matrix
import numpy as np

from logging import getLogger

logger = getLogger(__name__)


class TSP:

    def __init__(self, n_cities, city_coods):
        self.n_cities = n_cities
        self.city_coords = city_coods

        self.dist = distance_matrix(city_coods, city_coods)

        self.x = None
        self.u = None

    def solve(self):
        self.prepare()
        self._solve()
        return self.x_sol, self.u_sol

    def prepare(self):
        logger.debug('enter')
        self.model = LpProblem(f"TSP_{self.n_cities}", LpMinimize)
        self.make_val()

        for cnst in self.const_degree_from():
            self.model += cnst

        for cnst in self.const_degree_to():
            self.model += cnst

        for cnst in self.const_remove_partial_loop():
            self.model += cnst

        self.model += self.make_obj()

        # self.model.writeLP(f"model_{self.n_cities}.lp")
        with open(f'model_{self.n_cities}.pkl', 'wb') as f:
            pickle.dump(self, f, -1)
        logger.debug('exit')

    def _solve(self):
        logger.debug('enter')
        # self.model.solve(COINMP_DLL(
        #    mip=1, msg=1, cuts=1, presolve=1, dual=1, crash=0, scale=1, rounding=1, integerPresolve=1, strong=5, timeLimit=300, epgap=None
        # ))
        self.model.solve(COIN(msg=1,
                              threads=8,
                              maxSeconds=20
                              ))
        print("obj:", self.model.status)

        self.x_sol = np.array([list(map(pulp.value, row)) for row in self.x])
        self.u_sol = list(map(pulp.value, self.u))
        logger.debug('exit')

    def make_val(self):
        logger.debug('enter')
        self.x = np.array([[LpVariable("x_%s_%s" % (i, j), 0, 1,  # cat='Binary'
                                       ) if i != j else 0
                            for j in range(self.n_cities)] for i in range(self.n_cities)])
        self.u = np.array([LpVariable("u_%s" % (i), 0) for i in range(self.n_cities)])
        logger.debug('exit')

    def make_obj(self):
        logger.debug('enter')
        obj = lpSum([self.dist[i, j] for j in range(self.n_cities) for i in range(self.n_cities)])
        logger.debug('exit')
        return obj

    def const_degree_from(self):
        logger.debug('enter')
        consts = []
        for j in range(self.n_cities):
            consts.append((lpSum(self.x[:, j].tolist()) == 1, f'const_degree_from_{j}'))
        logger.debug('exit')
        return consts

    def const_degree_to(self):
        logger.debug('enter')
        consts = []
        for i in range(self.n_cities):
            consts.append((lpSum(self.x[i, :].tolist()) == 1, f'const_degree_to_{i}'))
        logger.debug('exit')
        return consts

    def const_remove_partial_loop(self):
        logger.debug('enter')
        consts = []
        for i in range(1, self.n_cities):
            for j in range(self.n_cities):
                if i != j:
                    consts.append((self.u[i] + 1 <= self.u[j] + self.n_cities *
                                   (1 - self.x[i, j]), f'const_remove_partial_loop_{i}_{j}'))
        logger.debug('exit')
        return consts


def main(filepath):
    df = pd.read_csv(filepath)
    tsp = TSP(df.shape[0], df[['X', 'Y']].values)
    x_sol, _ = tsp.solve()

    with open(filepath + '.x_sol.pkl', 'wb') as f:
        pickle.dump(x_sol, f, -1)
    with open(filepath + '.x_sol.pkl', 'rb') as f:
        x_sol = pickle.load(f)
    is_visited = np.zeros(df.shape[0], dtype=np.bool)
    idx = 0
    is_visited[idx] = True
    route = [0]
    while 1:
        points = x_sol[idx].argsort()[::-1]
        points = points[~is_visited[points]]
        if len(points) == 0:
            break
        idx = points[0]
        is_visited[idx] = True
        route.append(idx)
    route += [0]
    logger.debug(f'route: {route}')
    score = score_path(filepath, route)
    logger.info(f'score: {len(route)} {score}')
    pd.Series(route, name='Path').to_csv(filepath + '.lp.sub.csv', index=False)


if __name__ == '__main__':
    import pandas as pd
    import os
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
    main(filepath)
    #filepath = 'cities10000.csv'
    # main(filepath)
    #filepath = 'cities.csv'
    # main(filepath)
