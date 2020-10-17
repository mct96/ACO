import pandas as pd
import numpy as np
from numpy.random import RandomState
import argparse
import math
from copy import deepcopy
from datetime import datetime


class ACO:
    def __init__(self, graph, output, ants, iterations, initial_pheromone,
                 decay_rate, alpha, beta, xi, Q, elitism, elitism_gain, seed):
        self._graph = graph
        self._output = output
        self._ants = ants
        self._iterations = iterations
        self._initial_pheromone = initial_pheromone
        self._decay_rate = decay_rate
        self._alpha = alpha
        self._beta = beta
        self._xi = xi
        self._q = Q
        self._elitism = elitism
        self._eg = elitism_gain

        self._rd = RandomState(seed)
        self._pheromone = np.full(graph.shape, self._initial_pheromone)

        self._best_ant = None
        self._statistics = []

        for i in range(80):
            ants, max_cost = self._build_colony(40)
            
            local_best_ant, best_ant = ants[0], self._best_ant
            if not best_ant or local_best_ant[1] > best_ant[1]:
                self._best_ant = ants[0]

            self._update(ants, normalize_by=max_cost)
            

            
    def _build_colony(self, n):
        ants = []
        props = self._props()

        biggest_cost = 0
        for i in range(n):
            path, cost = self._build_path(props)
            total_cost = np.sum(cost)
            biggest_cost = np.max((total_cost, biggest_cost))
            ant = [path, total_cost]
            ants.append(ant)

        ants.sort(key=lambda ant: ant[1], reverse=True)
        return ants, biggest_cost
            

    def _build_path(self, probs):
        tabu_list = self._make_list(1)
        cost = self._make_list(-1)
        path = self._make_list(-1)

        pos, i = self._goto(shape=tabu_list.shape), 0
        while np.sum(tabu_list) >= 0:
            tabu_list[pos] = 0.0
            local_probs = tabu_list * probs[pos]
            den = np.sum(local_probs)
            if den == 0:
                break
            
            local_probs /= den
            goto = self._goto(probs=local_probs)

            cost[pos] = self._graph[pos][goto]
            path[i] = pos
            pos, i = goto, i+1

        path = path[np.where(path != -1)]
        cost = cost[np.where(cost != -1)]
        print(path.shape)
        return path, np.sum(cost)


    def _props(self):
        a, b = self._alpha, self._beta
        probs = self._pheromone ** a * self._graph ** b
        return probs


    def _make_list(self, v):
        n_nodes = self._graph.shape[0]
        lst = np.full(shape=n_nodes, fill_value=v)
        return lst


    def _goto(self, *, probs=None, shape=None):
        if np.any(probs) != None:
            return self._rd.choice(np.arange(probs.shape[0]), p=probs)
        elif np.any(shape) != None:
            return self._rd.choice(np.arange(shape[0]))

        
    def _update(self, ants, normalize_by):
        self._decrease_ph()

        for route, L in ants:
            self._update_ph(route, L/normalize_by)

        elitist_route = self._best_ant[0]
        elitist_cost = self._eg * self._best_ant[1]
        self._update_ph(elitist_route, elitist_cost/normalize_by)
        
        
    def _decrease_ph(self):
        p = self._decay_rate
        self._pheromone *= (1-p)

        
    def _update_ph(self, path, L):
        from_, to_ = path[:-1], path[1:]
        q = self._q
        route = np.column_stack((from_, to_))
        self._pheromone[route] += q * L



def load(filename):
    df = pd.read_csv(filename, sep="\t", header=None)
    nodes = np.unique((df.iloc[:, 0], df.iloc[:, 1]))
    rows = np.max(nodes)
    distances = np.zeros(shape=(rows, rows))
    from_= df.iloc[:, 0].to_numpy() - 1
    to_ = df.iloc[:, 1].to_numpy() - 1
    dists = df.iloc[:, 2].to_numpy()
    distances[from_, to_] = dists

    return distances


def main():
    aparse = argparse.ArgumentParser()
    aparse.add_argument("dataset",
                        nargs=1,
                        type=str,
                        help="Dataset with the graphs.")

    aparse.add_argument("output",
                        nargs=1,
                        type=str,
                        help="Save result")

    aparse.add_argument("ants",
                        nargs=1,
                        type=int,
                        help="Number of ants.")

    aparse.add_argument("iterations",
                        nargs=1,
                        type=int,
                        help="Number of iterations.")

    aparse.add_argument("-t", "--initial-pheromone",
                        nargs=1,
                        type=float,
                        default=[0.5],
                        help="Initial value for pheromone. Should be greater than 0.")

    aparse.add_argument("-d", "--decay-rate",
                        nargs=1,
                        type=float,
                        help="Decay rate of pheromone. Should be betweetn [0, 1].")

    aparse.add_argument("-a", "--alpha",
                        nargs=1,
                        type=float,
                        default=[1.0],
                        help="Paramenter alpha.")

    aparse.add_argument("-b", "--beta",
                        nargs=1,
                        type=float,
                        default=[5.0],
                        help="Parameter beta.")

    aparse.add_argument("--xi",
                        nargs=1,
                        type=float,
                        default=[1.0],
                        help="Backoff decay rate.")

    aparse.add_argument("-Q", "--reinforcement-gain",
                        nargs=1,
                        type=float,
                        default=[100],
                        help="Pheromone release reinforcement gain.")

    aparse.add_argument("-e", "--eletism",
                        action="store_const",
                        const=True,
                        default=False,
                        help="Eletism.")

    aparse.add_argument("-g", "--eletism-gain",
                        nargs=1,
                        type=float,
                        default=[5.0],
                        help="Gain of eletist ants")

    args = aparse.parse_args()
    dataset = args.dataset[0]
    output = args.output[0]
    ants = args.ants[0]
    iterations = args.iterations[0]
    initial_pheromone = args.initial_pheromone[0]
    decay_rate = args.decay_rate[0]
    alpha = args.alpha[0]
    beta = args.beta[0]
    xi = args.xi[0] # control stochastic backoff
    reinforcement_gain = args.reinforcement_gain[0]
    eletism = args.eletism
    eletism_gain = args.eletism_gain[0]

    parameters = f"dataset: {dataset}\n"\
                 f"output: {output}\n"\
                 f"ants: {ants}\n"\
                 f"iterations: {iterations}\n"\
                 f"initial pheromone: {initial_pheromone}\n"\
                 f"decay rate: {decay_rate}\n"\
                 f"alpha: {alpha}\n"\
                 f"beta: {beta}\n"\
                 f"xi: {xi}\n"\
                 f"reinforcement gain: {reinforcement_gain}\n"\
                 f"eletism: {eletism}\n"\
                 f"eletism gain: {eletism_gain}"

    print(parameters)
    graph = load(dataset)

    aco = ACO(graph, output, ants, iterations, initial_pheromone,
              decay_rate, alpha, beta, xi, reinforcement_gain, eletism,
              eletism_gain, 0)


if __name__ == "__main__":
    main()
