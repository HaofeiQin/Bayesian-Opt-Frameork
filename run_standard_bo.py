import time
import json
import numpy as np
import math
from ax.modelbridge import get_sobol, get_GPEI
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient
from framework_problem import ContextualRunner, Agent, arrival_aggregate_reward

########
# Define problem
NUM_USER = 3
NUM_CREATOR = 3
num_contexts = 9
num_trials = 100
seg_i = [1, 1, 1]
seg_j = [1,1 , 1]

#seg_i = [1 / 2, 1 / 3, 1 / 6]
#seg_j = [1 / 6, 1 / 2, 1 / 3]

theta_i = np.random.uniform(low=0, high=1, size=5)
theta_j = np.random.uniform(low=0, high=1, size=5)
x_i=np.random.uniform(low=-1, high=1, size=5)
x_j=np.random.uniform(low=-1, high=1, size=5)

num_contexts = 9

# Context_List=[seg_i, seg_j, num_i, num_j]

#theta [0,1], x[-1,1]
lambda_i = [30,30,30]
lambda_j = [30,30,30]
lambda_all=[]

for i in range(6):
    Context_List = [seg_i, seg_j,lambda_i, lambda_j]
    Old_Lambda=lambda_i+lambda_j


    benchmark_problem = ContextualRunner(
        num_contexts=num_contexts,
        context_list=Context_List
    )

    gs = GenerationStrategy(
        name="GPEI",
        steps=[
            GenerationStep(get_sobol, 8),
            GenerationStep(get_GPEI, -1),
        ],
    )
    axc = AxClient(generation_strategy=gs)

    experiment_parameters = benchmark_problem._contextual_parameters
    axc.create_experiment(
        name="aggregated_reward_experiment",
        parameters=experiment_parameters,
        objective_name="aggregated_reward",
        minimize=False,
        overwrite_existing_experiment=True,
    )


    def evaluation_aggregated_reward(parameters):
        x = []
        for value in parameters.values():
            x.append(value)
        aggregated_reward = benchmark_problem.Total_Reward(Rho_ij=x, Old_lambda=Old_Lambda)
        return {"aggregated_reward": (aggregated_reward, 0.0)}


    for itrial in range(num_trials):
        parameters, trial_index = axc.get_next_trial()
        aggregated_res = evaluation_aggregated_reward(parameters)
        axc.complete_trial(trial_index=trial_index, raw_data=aggregated_res)

    best_parameters = axc.get_best_parameters()
    Rho_ij = []
    for value in best_parameters.values[0]():
        Rho_ij.append(value)


    lambda_i = benchmark_problem.f_i(Rho_ij)
    lambda_j = benchmark_problem.f_j(Rho_ij)