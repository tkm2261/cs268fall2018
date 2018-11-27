# https://github.com/jvkersch/pyconcorde
from concorde.tsp import TSPSolver
import pandas as pd
import numpy as np
from utils import score_path, score_path_santa


def make_submission(name, path):
    #assert path[0] == path[-1] == 0
    #assert len(set(path)) == len(path) - 1 == 197769
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)


def concorde(filename):
    seed = 0

    cities = pd.read_csv(filename)
    solver = TSPSolver.from_data(cities.X, cities.Y, norm="EUC_2D")

    tour_data = solver.solve(time_bound=1800.0, verbose=True, random_seed=seed)
    if tour_data.found_tour:
        path = np.append(tour_data.tour, [0])
        make_submission(filename + '.path.csv', path)
        return path
    else:
        return None


if __name__ == '__main__':
    #path = concorde('cities1000.csv')
    #print('score: ', score_path('cities1000.csv', path))

    #path = concorde('cities10000.csv')
    #print('score: ', score_path('cities10000.csv', path))

    path = concorde('cities.csv')
    print('score: ', score_path('cities.csv', path))
