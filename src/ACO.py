import pandas as pd
import numpy as np
from numpy.random import RandomState
import argparse
import math
from copy import deepcopy
from datetime import datetime
from scipy.stats import sem, t
from datetime import datetime




class ACO:
    def __init__(self, graph, ants, iterations, initial_pheromone, decay_rate,
                 alpha, beta, xi, Q, elitism, elitism_gain, seed):
        self._graph = graph
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
        self._pheromone = None

        self._best_ant = None
        self._statistics = None


    def _save_statistics(self, costs):
        max_v = np.max(costs)
        min_v = np.min(costs)
        mean = np.mean(costs)
        std = np.std(costs)
        median = np.median(costs)

        statistics = [max_v, min_v, mean, std, median]
        self._statistics.append(statistics)


    def set_seed(self, seed):
        self._rd = RandomState(seed)


    def _reset_state(self):
        self._pheromone = np.full(self._graph.shape, self._initial_pheromone)
        self._statistics = list()


    def fit(self):
        self._reset_state()

        for i in range(self._iterations):
            ants, max_cost = self._build_colony(self._ants)
            costs = np.array([ant[1] for ant in ants])
            self._save_statistics(costs)

            local_best_ant, best_ant = ants[0], self._best_ant
            if not best_ant or local_best_ant[1] > best_ant[1]:
                self._best_ant = ants[0]

            self._update(ants, normalize_by=max_cost)
        return self._best_ant


    def _build_colony(self, n):
        ants = list()
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
        cost = self._make_list(0)
        path = self._make_list(-1)

        pos, i = self._goto(shape=tabu_list.shape), 1
        path[0] = pos
        while np.sum(tabu_list) > 0:
            tabu_list[pos] = 0.0
            local_probs = tabu_list * probs[pos]
            den = np.sum(local_probs)
            if den == 0:
                break

            local_probs /= den
            goto = self._goto(probs=local_probs)

            cost[i] = self._graph[pos][goto]
            path[i] = goto
            pos, i = goto, i+1

        path = path[np.where(path != -1)]
        return path, cost


    def _props(self):
        a, b = self._alpha, self._beta
        probs = (self._pheromone ** a) * (self._graph ** b)
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

        if self._elitism:
            elitist_route = self._best_ant[0]
            elitist_cost = self._eg * self._best_ant[1]
            self._update_ph(elitist_route, elitist_cost/normalize_by)


    def _decrease_ph(self):
        p = self._decay_rate
        self._pheromone *= (1-p)


    def _update_ph(self, path, L):
        from_, to_ = path[:-1], path[1:]
        q = self._q
        self._pheromone[from_, to_] += q * L




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

    aparse.add_argument("-e", "--elitism",
                        action="store_const",
                        const=True,
                        default=False,
                        help="Enable elitism.")

    aparse.add_argument("-g", "--elitism-gain",
                        nargs=1,
                        type=float,
                        default=[5.0],
                        help="Gain of elitist ants")

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
    elitism = args.elitism
    elitism_gain = args.elitism_gain[0]

    parameters = "PARAMETERS".center(80, " ") + "\n\n"
    parameters += f"dataset: {dataset}\n"\
                  f"output: {output}\n"\
                  f"ants: {ants}\n"\
                  f"iterations: {iterations}\n"\
                  f"initial pheromone: {initial_pheromone}\n"\
                  f"decay rate: {decay_rate}\n"\
                  f"alpha: {alpha}\n"\
                  f"beta: {beta}\n"\
                  f"xi: {xi} (no effect yet)\n"\
                  f"reinforcement gain: {reinforcement_gain}\n"\
                  f"elitism: {elitism}\n"\
                  f"elitism gain: {elitism_gain}"

    print(parameters)
    print("-" * 80)
    parameters += "\n" + "-" * 80 + "\n"
    parameters += "PERFORMANCE".center(80, " ") + "\n\n"
    graph = load(dataset)

    aco = ACO(graph, ants, iterations, initial_pheromone,
              decay_rate, alpha, beta, xi, reinforcement_gain, elitism,
              elitism_gain, seed=0)

    statistics = list()
    for i in range(5):
        print(f"replication {i} started. {' ' * 15}", end="\r")
        seed = abs(int(13 * i + 11 * i + 7 * i + 5 * i + 3 * i + 2 * i + i + 1))
        aco.set_seed(seed=seed)
        start = datetime.now()
        solution = aco.fit()
        end = datetime.now()
        statistics.append(aco._statistics)
        parameters += f"\nseed {seed}, replication took: {end-start}"
    print("✔ - replications completed. ")

    result = statistical_report(np.array(statistics))
    save_results(output, result)
    save_parameters(output, parameters)




def statistical_report(replications, confidence=0.95):
    statistics = np.mean(replications, axis=0)
    rows = statistics.shape[0]
    stderr = sem(replications, axis=0)
    n = rows
    h = stderr * t.ppf((1+confidence)/2, n - 1)
    statistics = np.append(statistics, h, axis=1)

    return statistics




def save_results(output, result):
    print("saving results...", end="\r")
    df = pd.DataFrame(data=result,
                      columns=["max", "min", "mean", "std", "median",
                               "stderr (max)", "stderr (min)", "stderr (mean)",
                               "stderr (std)", "stderr (median)"])

    df.to_csv(output, float_format="%.4f", index=False)
    print("✔ - results saved! :D")




def save_parameters(output, parameters):
    print("saving parameters...", end="\r")
    base_name = "".join(output.rsplit(".")[:-1])
    base_name = f"{base_name}_params.txt"
    with open(base_name, "wt", encoding="utf-8") as f:
        f.write(parameters + "\n")
    print("✔ - parameters saved! xD")




if __name__ == "__main__":
    main()
