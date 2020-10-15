import pandas as pd
import numpy as np
from numpy.random import RandomState
import argparse
import math
from copy import deepcopy
from datetime import datetime
from scipy.sparse import csr_matrix

class ACO:
    def __init__(self, graph, nodes, ants, iterations, initial_pheromone,
                 decay_rate, alpha, beta, xi, Q, elitism, elitism_gain, output, seed):
        self._graph = graph
        self._nodes = nodes
        self._ants = ants
        self._iterations = iterations
        self._initial_pheromone = initial_pheromone
        self._decay_rate = decay_rate
        self._alpha = alpha
        self._beta = beta
        self._xi = xi
        self._Q = Q
        self._EG = elitism_gain
        self._elitism = elitism
        self._output = output
        print(output)

        self._rd = RandomState(seed)
        self._pheromone = csr_matrix(1, self._graph.shape)
        
        self._best_ant = None
        self._statistics = []


    def fit(self):
        self._explore()

    def _explore(self):
        ants = self._build_ants()


        
    def _release_eletist(self):
        pass


    def _decrease_pheromone(self):
        pass

    
    def _probabilities(self):
        A = self._pheromone.power(self._alpha)
        B = self._graph.power(self._beta)
        AB = A.multiply(B)
        print(AB.toarray())
        N = 1/np.array(AB.sum(axis=1))
        N[N==np.inf] = 1
        print(np.min(N))
        AB = AB.divide(1/N)
        
        return AB

    def _build_ants(self):
        probabilities = self._probabilities()
    

    def _build_path(self):
        tabu_list, path, path_cost, src = self._init_path()

        backoff, folder = 1, []
        while len(tabu_list):
            possibilities = self._graph[src]["to"]
            cost = self._graph[src]["cost"]
            probs = self._choice_destiny(src, tabu_list, possibilities, cost)

            if np.any(np.isnan(probs)):
                self._store_in_folder(folder, path, path_cost)

                if not self._backoff_again(backoff) or backoff >= len(path):
                    return self._best_of_folder(folder)

                tabu_list.extend(path[-backoff:])
                path = path[:-backoff]
                path_cost = path_cost[:-backoff]
                src = path[-1]
                backoff += 1
                continue

            src = self._rd.choice(possibilities, p=probs)
            idx = np.where(possibilities == src)

            path_cost.append(float(cost[idx]))
            path.append(src)
            tabu_list.remove(src)

        total_cost = sum(path_cost)
        return path, path_cost, total_cost, len(list(set(path)))


    def _choice_destiny(self, src, tabu_list, possibilities, cost):
        distances = list()
        prob = np.zeros(possibilities.shape)
        a, b = self._alpha, self._beta
        i = 0
        pheromones = np.array(self._pheromone[src]) ** a
        for p, d in zip(possibilities, cost):
            if p in tabu_list:
                local_pheromone = pheromones[i]
                prob[i] = local_pheromone * self._visibility(d, b)
            i += 1

        den = np.sum(prob)
        if den:
            return prob/den
        else:
            return np.nan

    def _reset_ant(self):
        tabu_list = list(self._nodes)
        path = list()
        cost = list()
        return tabu_list, path, cost

def load(filename):
    df = pd.read_csv(filename, sep="\t", header=None)
    from_ = df.iloc[:, 0].to_numpy(dtype=np.int)
    to_ = df.iloc[:, 1].to_numpy(dtype=np.int)
    dists = df.iloc[:, 2].to_numpy()
    distances = csr_matrix((dists, (from_, to_)))
    nodes = np.unique((from_, to_))
    return distances, nodes

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

    aparse.add_argument("-q", "--reinforcement-gain",
                        nargs=1,
                        type=float,
                        default=[100],
                        help="Pheromone release reinforcement gain.")

    aparse.add_argument("-e", "--elitism",
                        action="store_const",
                        const=True,
                        default=False,
                        help="Elitism.")

    aparse.add_argument("-g", "--elitism-gain",
                        nargs=1,
                        type=float,
                        default=[5.0],
                        help="Gain of elitist ants")

    args = aparse.parse_args()
    print(args)
    dataset = args.dataset[0]
    output = args.output[0]
    ants = args.ants[0]
    iterations = args.iterations[0]
    initial_pheromone = args.initial_pheromone[0]
    decay_rate = args.decay_rate[0]
    alpha = args.alpha[0]
    beta = args.beta[0]
    xi = args.xi[0] # control stochastic backoff
    Q = args.reinforcement_gain[0]
    elitism = args.elitism
    elitism_gain = args.elitism_gain[0]

    parameters = f"dataset: {dataset}\n"\
                 f"output: {output}\n"\
                 f"ants: {ants}\n"\
                 f"iterations: {iterations}\n"\
                 f"initial pheromone: {initial_pheromone}\n"\
                 f"decay rate: {decay_rate}\n"\
                 f"alpha: {alpha}\n"\
                 f"beta: {beta}\n"\
                 f"xi: {xi}\n"\
                 f"reinforcement gain: {Q}\n"\
                 f"elitism: {elitism}\n"\
                 f"elitism gain: {elitism_gain}"

    print(parameters)

    graph, nodes = load("../dataset/entrada3.txt")

    statistics = []
    for i in range(5):
        print(f"replication: {i+1}/5")
        i = int(abs(math.sin(i) * 1000))
        aco = ACO(graph, nodes, ants, iterations, initial_pheromone,
                  decay_rate, alpha, beta, xi, Q, elitism,
                  elitism_gain, output, i)

        pop = aco.fit()
        exit()
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
