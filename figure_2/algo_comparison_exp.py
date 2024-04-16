import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from itertools import product
from multiprocessing import cpu_count, Pool, Manager

from dataclasses import dataclass
from collections import defaultdict
import abc
from typing import Tuple
import random as rd

def save_data(directory_path,file_name, results):
    data_by_key = defaultdict(list)
    for tup in results: 
        data_by_key[tup[0]].append(tup[1:])
    file_paths = {}
    for key, values in data_by_key.items():
        file_path = f"{directory_path}/{file_name}.csv"
        with open(file_path, 'w') as file:
            for value in values:
                file.write(f"{value[0]}, {value[1]}, {value[2]}\n")
        file_paths[key] = file_path

class Distorted_OneMax(abc.ABC):
    def __init__(self, distortion_probability: float, distortion: float, dim: int):
        self._distortion_probability = distortion_probability
        self._distortion = distortion
        self._dim = dim
        self.point_dict = {}
    
    def __call__(self, search_point: np.ndarray) -> float:
        point_hash = hash(search_point.tobytes())
        if point_hash not in self.point_dict:
            onemax_value = np.count_nonzero(search_point)
            self.point_dict[point_hash] = onemax_value
            if rd.random() < self._distortion_probability:
                self.point_dict[point_hash] += self._distortion
        #else:
            #print("seen")
        return self.point_dict[point_hash]

class Algorithm(abc.ABC):
    def __init__(self, budget, target_k, elitist, init_lam, prob_mutation):
        self._budget = budget
        self.target_k = target_k
        self._elitist = elitist
        self._init_lam = init_lam
        self._prob_mutation = prob_mutation

        self.total_evals = 1
        self.cur_searchpoint: np.array = None
        self.cur_lam: float = self._init_lam
        self.cur_distorted: bool = False
        self.cur_fitness = float('-inf')
        self.cur_onemax = float('-inf')


    def __call__(self, problem: Distorted_OneMax):
        dim = problem._dim
        fitness_func = problem

        # Initialize search point
        initial_searchpoint =np.random.choice([0, 1], dim)
        initial_fitness = fitness_func(initial_searchpoint)
        self.update_parent(searchpoint=initial_searchpoint,searchpoint_fitness=initial_fitness)

        while not self.should_terminate(problem, int(np.round(self.cur_lam))):
            #Update total number of evaluations
            self.total_evals += int(np.round(self.cur_lam))
            #Mutation
            offsprings, offsprings_fitness = self.mutate(fitness_func=fitness_func, dim = dim)
            #Selection
            selected_offspring, selected_offspring_fitness = self.select(offsprings=offsprings, offsprings_fitness=offsprings_fitness)
            #Update algorithm
            self.update_algo(selected_offspring, selected_offspring_fitness)
            #Update search point
            self.update_parent(searchpoint=selected_offspring, searchpoint_fitness=selected_offspring_fitness)
        return self.total_evals

    def update_parent(self, searchpoint, searchpoint_fitness):
        self.cur_searchpoint = searchpoint
        self.cur_fitness = searchpoint_fitness
        onemax_value = np.count_nonzero(self.cur_searchpoint)
        self.cur_distorted = (onemax_value != self.cur_fitness)
        self.cur_onemax = onemax_value

    def reset(self):
        self.total_evals = 1
        self.cur_searchpoint: np.ndarray = None
        self.cur_lam: float = self._init_lam
        self.cur_distorted: bool = False
        self.cur_fitness = float('-inf')
        self.cur_onemax = float('-inf')
    
    def mutate(self, fitness_func: Distorted_OneMax, dim) -> Tuple[np.ndarray, np.ndarray]:
        # If current lambda is a float round to nearest int
        num_offspring = int(np.round(self.cur_lam))
        
        #Take the parent as offspring if elitist
        size = num_offspring+1 if self._elitist else num_offspring
        offsprings = np.empty((size, dim), dtype=int)
        offsprings_fitness = np.empty(size)
        if self._elitist:
            offsprings[-1] = self.cur_searchpoint
            offsprings_fitness[-1] = self.cur_fitness

        for i in range(num_offspring):
            offsprings[i] = self.cur_searchpoint
            # how manny bits are flipped in a single offspring.
            number_bits_flipped = np.random.binomial(dim, self._prob_mutation)
            #choose positions to be flipped
            idxs = np.random.choice(dim, number_bits_flipped, False, p= (np.ones(dim) / dim))
            #Flip those bits of the parent
            offsprings[i,idxs] ^= 1
            offsprings_fitness[i] = fitness_func(offsprings[i])
        return offsprings, offsprings_fitness
    
    def select(self, offsprings: np.ndarray, offsprings_fitness: np.ndarray) -> Tuple[np.ndarray, float]:
        max_offspring_fitness = np.max(offsprings_fitness)
        selected_offspring_idx = np.random.choice(np.where(offsprings_fitness == max_offspring_fitness)[0])
        return offsprings[selected_offspring_idx], max_offspring_fitness

    def update_algo(self, selected_offspring, selected_offspring_fitness):
        pass

    def should_terminate(self, problem: Distorted_OneMax, n_evals: int = 0) -> bool:
        if (self.total_evals + n_evals) > self._budget:
            print("OUT OF BUDGET")
        #Terminate if either the target fitness has been reached, or the next generation would exceed the budget
        return (
            self.cur_fitness >= (problem._dim - self.target_k)
            or (self.total_evals + n_evals) > self._budget
        )

@dataclass
class one_lambda(Algorithm):
    def __init__(self, 
                 budget: int = None,
                 target_k: int = None,
                 elitist: bool = None,
                 init_lam: float = 1,
                 prob_mutation: float = None):
        
        super().__init__(budget=budget, target_k= target_k, elitist=elitist,init_lam= init_lam,prob_mutation= prob_mutation)

@dataclass
class SEAR(Algorithm):
    def __init__(self, 
                 budget: int = None,
                 target_k: int = None,
                 elitist: bool = None,
                 init_lam: float = 1,
                 prob_mutation: float = None,
                 f_factor: float = None,
                 s_factor: float = None,
                 lam_max: int = None):
        
        super().__init__(budget=budget, target_k= target_k, elitist=elitist,init_lam= init_lam,prob_mutation= prob_mutation)
        self._f_factor = f_factor
        self._s_factor = s_factor
        self._lam_max = lam_max

    def update_algo(self, selected_offspring, selected_offspring_fitness):
        #Update Algorithm Parameters
        if selected_offspring_fitness > self.cur_fitness:
            self.cur_lam = self.cur_lam / self._f_factor
        else:
            if self.cur_lam == self._lam_max:
                self.cur_lam = 1
            else:
                self.cur_lam = self.cur_lam * (self._f_factor ** (1/self._s_factor))
        self.cur_lam = np.clip(self.cur_lam, 1, self._lam_max)

    
def run_optimizer(temp):
    logging_list, algorithm, budget, target_k, dim, distortion, p_distortion, f_factor, s_factor, lam_max, lam_com_plus = temp
    prob_mutation = 1/dim
    #choose the correct algorithm
    alg = None
    if algorithm == "SEAR": alg= SEAR(budget=budget, target_k= target_k, elitist = False, init_lam = 1, prob_mutation =prob_mutation, f_factor = f_factor,s_factor = s_factor, lam_max = lam_max)
    elif algorithm == "OLEA": alg = one_lambda(budget=budget, target_k= target_k, elitist = False, init_lam = lam_com_plus, prob_mutation =prob_mutation)
    else: alg = one_lambda(budget=budget, target_k= target_k, elitist = True, init_lam = lam_com_plus, prob_mutation =prob_mutation)

    f_instance = Distorted_OneMax(distortion_probability= p_distortion, distortion= distortion, dim=dim)
    run_result = alg(f_instance)
    logging_list.append((algorithm, dim, p_distortion, run_result))

def run_parallel(args, file_name):
    with Pool(min(cpu_count(), len(args))) as pool:
        print(pool)
        shared_list = Manager().list()
        pool.map(run_optimizer, [(shared_list,) + arg for arg in args])
    directory_path = 'output_files'
    save_data(directory_path, file_name, sorted(shared_list, key=lambda x: (x[0], x[1], x[2])))

def set_params(algorithms, budget, target_k, runs, dim, distortion, p_distortion,f_factor, s_factor, lam_max, lam_com_plus):
    return list(product(algorithms, budget, target_k, np.repeat(dim, runs), distortion, p_distortion, f_factor, s_factor, lam_max, lam_com_plus))
    
def exp_v1(input):
    n_values= np.arange(60, 520, 20)
    args = []
    for n in n_values:
        eta = np.e/(np.e-1)
        lam_com_plus = 1.5 *np.log(n)
        p_distortion = eta**(-lam_com_plus)
        args.extend(set_params(algorithms = input,
                               budget = [1000000],
                               target_k = [n**0.4],
                               runs = 50,
                               dim = n,
                               distortion = [np.log(n)],
                               p_distortion = [p_distortion],
                               f_factor = [1.5],
                               s_factor = [1],
                               lam_max = [n*np.log(n)],
                               lam_com_plus = [lam_com_plus]))
    return args

if __name__ == '__main__':
    #Get all parameter combinations
    run_parallel(exp_v1(["OLEA"]), 'OLEA_values')
    run_parallel(exp_v1(["OPEA"]), 'OPEA_values')
    run_parallel(exp_v1(["SEAR"]), 'SEAR_values')