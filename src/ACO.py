import pandas as pd
import numpy as np
from numpy.random import RandomState
import argparse
import math
from copy import deepcopy
from datetime import datetime

class ACO:
    def __init__(self, graph, output, ants, iterations, initial_pheromone,
                 decay_rate, alpha, beta, xi, Q, eletism, eletism_gain, seed):
        self._vertices = list(graph.keys())
        self._n_vertices = len(graph)
        self._n_edges = sum([edges["to"].size for vertice, edges \
                                              in graph.items()])
        self._graph = graph
        self._output = output
        self._ants = ants
        self._iterations = iterations
        self._initial_pheromone = initial_pheromone
        self._decay_rate = decay_rate
        self._alpha = alpha
        self._beta = beta
        self._xi = xi
        self._Q = Q
        self._EG = eletism_gain
        self._eletism = eletism

        self._rd = RandomState(seed)
        self._pheromone = self._init_pheromone(zero=False)

        self._best_ant = None
        self._statistics = []


    def fit(self):
        for i in range(self._iterations):
            print(f"iteration: {i+1}/{self._iterations}{' '*20}")
            ants = self._explore()
            costs = np.array([ant[0] for ant in ants])
            max_cost = np.max(costs)
            min_cost = np.min(costs)
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            median_cost = np.median(costs)
            self._statistics.append(np.array([max_cost, min_cost, mean_cost,
                                              std_cost, median_cost]))

        print()
        return np.array(self._statistics)


    def _explore(self):
        ants = self._build_ants()

        best_local_ant = ants[0]
        if self._best_ant == None or best_local_ant[0] > self._best_ant[0]:
            self._best_ant = best_local_ant

        delta = self._release_pheromone(ants)
        decreased_pheromone = self._decrease_pheromone()
        if self._eletism:
            eletist_ant = self._release_eletist()

        for _from, pheromone in self._pheromone.items():
            if self._eletism:
                self._pheromone[_from] = decreased_pheromone[_from] +\
                    delta[_from] + self._EG * eletist_ant[_from]
            else:
                self._pheromone[_from] = decreased_pheromone[_from] +\
                    delta[_from]

        return ants


    def _init_pheromone(self, zero):
        pheromone = dict()
        for node, edges in self._graph.items():
            to = edges["to"]
            cost = edges["cost"]

            if zero:
                pheromone[node] = np.zeros_like(to, dtype=np.float)
            else:
                pheromone[node] = np.full(to.shape, self._initial_pheromone,
                                          dtype=np.float)

        return pheromone


    def _release_pheromone(self, ants):
        d_ph = self._init_pheromone(zero=True) # initialize with zeros.

        max_value = np.array(0)
        for ant in ants:
            total_cost, path = ant
            delta = self._Q * total_cost # released pheromone by ant "ant".
            for _from, _to in zip(path[:-1], path[1:]): # for each edge.
                to = self._graph[_from]["to"] # all neighbors of _from.
                idx_dst = np.where(to == _to) # index in pheromone vector.
                d_ph[_from][idx_dst] += delta

                max_value = np.max([max_value, np.max(d_ph[_from])])

        for _from, pheromone in d_ph.items():
            d_ph[_from] /= max_value

        return d_ph


    def _release_eletist(self):
        d_ph = self._init_pheromone(zero=True)

        path = self._best_ant[1]
        max_value = np.array(0)
        delta = self._Q * self._best_ant[0]
        for _from, _to in zip(path[:-1], path[1:]):
            to = self._graph[_from]["to"] # all neighbors of _from.
            idx_dst = np.where(to == _to) # index in pheromone vector.
            d_ph[_from][idx_dst] += delta

            max_value = np.max([max_value, np.max(d_ph[_from])])

        for _from, pheromone in d_ph.items():
            d_ph[_from] /= max_value

        return d_ph


    def _decrease_pheromone(self):
        d_ph = deepcopy(self._pheromone)
        p = self._decay_rate

        for _from, pheromone in d_ph.items():
            d_ph[_from] = (1-p) * pheromone

        return d_ph

    def _probabilities(self):
        a, b = self._alpha, self._beta
        probs = dict()
        for from_, pheromone in self._pheromone:
            T = pheromone ** a # Pheromone of all edges
            D = self._graph[from_][1] ** b # Length of all edges
            probs = T * D # TODO test if it is correct

        return probs

    
    def _build_ants(self):
        ranking = []
        probs = self._probabilites()
        
        for i in range(self._ants):
            print(f"ant {i+1}/{self._ants}{' '*20}", end="\r")
            path, path_cost, total_cost, n = self._build_path()
            ranking.append((total_cost, path))# , path_cost, n))

        print()
        ranking.sort(reverse=True)
        return ranking


    def _init_path(self):
        tabu_list = list(self._vertices)
        src = self._rd.choice(tabu_list)
        tabu_list.remove(src)
        path = [src]
        path_cost = [0.0]
        return tabu_list, path, path_cost, src

    
    def _backoff_again(self, backoff):
        v = self._rd.random_sample() # a number in [0, 1)
        xi = self._xi
        return v < math.exp(-abs(xi*(backoff - 1)))


    def _best_of_folder(self, folder):
        folder.sort(reverse=True) # folder is a list with cost and path as
                                  # members.
        return folder[0]


    def _store_in_folder(self, folder, path, path_cost):
        path = deepcopy(path)
        path_cost = deepcopy(path_cost)
        total_cost = np.sum(path_cost)
        size = len(list(set(path)))
        folder.append([path, path_cost, total_cost, size])


    def _build_path(self, probs):
        tabu_list, path, path_cost, src = self._init_path()

        backoff, folder = 1, []
        while len(tabu_list):
            possibilities = self._graph[src]["to"]
            cost = self._graph[src]["cost"]
            dst_probs = self._next_move(src, probs, tabu_list)

            if np.any(np.isnan(dst_probs)):
                self._store_in_folder(folder, path, path_cost)

                if not self._backoff_again(backoff) or backoff >= len(path):
                    return self._best_of_folder(folder)

                tabu_list.extend(path[-backoff:])
                path = path[:-backoff]
                path_cost = path_cost[:-backoff]
                src = path[-1]
                backoff += 1
                continue

            dst = self._rd.choice(possibilities, p=dst_probs)
            idx = np.where(possibilities == dst)

            path_cost.append(float(cost[idx]))
            path.append(dst)
            tabu_list.remove(dst)
            src = dst

        total_cost = sum(path_cost)
        return path, path_cost, total_cost, len(list(set(path)))

    
    def _choice_destiny(self, src, probs, tabu_list):
        pass

    
def load(filename):
    df = pd.read_csv(filename, sep="\t", header=None)
    groups = df.groupby(by=0)
    graph = dict()
    for key, group in groups:
        to = group.iloc[:, 1].to_numpy()
        cost = group.iloc[:, 2].to_numpy()
        graph[key] = {"to": to, "cost": cost}

    return graph

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
    statistics = []
    for i in range(5):
        print(f"replication: {i+1}/5")
        i = int(abs(math.sin(i) * 1000))
        aco = ACO(graph, output, ants, iterations, initial_pheromone,
                  decay_rate, alpha, beta, xi, reinforcement_gain, eletism,
                  eletism_gain, i)

        pop = aco.fit()

        statistics.append(pop)

    statistics = np.mean(statistics, axis=0)

    df = pd.DataFrame(data=statistics,
                      columns=["max", "min", "mean", "std", "median"],
                      dtype=np.float)

    df.to_csv(output, index=False, float_format="%.4f")

    parameters_file = "".join(output.rsplit(".", 1)[:-1]) + "-params.txt"
    with open(parameters_file, "wt", encoding="utf-8") as f:
        f.write(parameters + "\n")


if __name__ == "__main__":
    main()
