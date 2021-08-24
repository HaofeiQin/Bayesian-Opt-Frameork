from typing import Dict, List
import numpy as np
import math

NUM_RUNS = 400
NUM_USER = 3
NUM_CREATOR = 3
'''
god: theta
observed: seg, x
opt: rho_ij
'''


class Agent(object):
    def __init__(
            self,
            seg_i: list = True,
            seg_j: list = True,
            num_i: list = True,
            num_j: list = True,
            rho_ij: list = True,

    ):
        """constructor.

        Args:
            seg_i:  a list proportion of user type i (i=1,2,3)
            seg_j:  a list proportion of creator type j (j=1,2,3)
            num_i: a list of user type i number (i=1,2,3)
            num_j: a list of creator type j number (j=1,2,3)
            rho_ij: a list proportion of creator type j recommended to user type i (list=[rho_11, rho_12, rho_13, rho_21,..., rho_32, rho_33])


        """
        self.seg_i = seg_i
        self.seg_j = seg_j
        self.num_i = num_i
        self.num_j = num_j
        self.rho_ij = rho_ij

    # cacluate the new arrival rate of user type i and creator type j
    def user_arrival_rate(self, i):
        List_i = np.array(self.rho_ij)
        List_i = (List_i.reshape(3, 3))
        for k in range(3):
            List_i [k, :]=List_i[k, :] / (np.sum(List_i[k, :]))
        List_i = List_i[i-1, :]
        f_i = np.dot(List_i, np.array(self.num_j))
        return f_i

    def creator_arrival_rate(self, j):
        List_j = np.array(self.rho_ij)
        List_j = (List_j.reshape(3, 3))
        for k in range(3):
            List_j [k, :]=List_j[k, :] / (np.sum(List_j[k, :]))
        List_j = List_j[:, j-1]
        f_j = np.dot(List_j, np.array(self.num_i))
        return f_j


class ContextualRunner:
    def __init__(self, num_contexts, context_list, return_context_reward=True):
        # context_list = [seg_i, seg_j, num_i, num_j, lambda_i, lambda_j]

        self.context_list = context_list
        self.return_context_reward = return_context_reward
        self.Seg_i = self.context_list[0]
        self.Seg_j = self.context_list[1]
        self.lambda_i = self.context_list[2]
        self.lambda_j = self.context_list[3]
        self.num_i = self.lambda_i+np.random.normal(0, 1,size=3)
        self.num_j = self.lambda_j+np.random.normal(0, 1,size=3) #simulation
        # num_i = np.multiply(seg_i, np.random.poisson(lam=self.lambda_i))
        # num_j = np.multiply(seg_j, np.random.poisson(lam=self.lambda_j))
        
        self.num_contexts = num_contexts
        self._contextual_parameters = []
        for i in range(NUM_USER):
            self._contextual_parameters.extend(
                [
                    {
                        "name": f"Rho_{i}{j}",
                        "type": "range",
                        "bounds": [0.0, 1.0],
                        "value_type": "float",
                        "log_scale": False,
                    }
                    for j in range(NUM_CREATOR )
                ]
            )

    def Total_Reward(self, Rho_ij, Old_lambda):
        agent_sample = Agent(
            seg_i=self.Seg_i,
            seg_j=self.Seg_j,
            num_i=self.num_i,
            num_j=self.num_j,
            rho_ij=Rho_ij,
        )
        context_rewards = arrival_aggregate_reward(
            agent=agent_sample, i=NUM_USER, j=NUM_CREATOR, old_lambda= Old_lambda
        )
        return context_rewards  # reward maximization
    def f_i(Rho_ij):
        new_agent = Agent(self.seg_i, self.seg_j, self.num_i, self.num_j, Rho_ij)
        lambda_i = []
        for i in range(NUM_USER):
            lambda_i.append(math.floor(np.dot(theta_i, x_i)+new_agent.user_arrival_rate(i+1)))
    
    def f_j(Rho_ij):
        new_agent = Agent(self.seg_i, self.seg_j, self.num_i, self.num_j, Rho_ij)
        lambda_i = []
        for i in range(NUM_USER):
            lambda_j.append(math.floor(np.dot(theta_j, x_j)+new_agent.creator_arrival_rate(j+1))



def arrival_aggregate_reward(agent, i, j, old_lambda):
    reward = 0
    for k in range(i):
        reward += agent.user_arrival_rate(k+1)
    for k in range(j ):
        reward += agent.creator_arrival_rate(k+1)

    # 负项，保证lambda值增加
    new_lambda = []
    for i in range(i):
        new_lambda.append(np.dot(theta_i, x_i) + agent.user_arrival_rate(i+1))
    for j in range(j):
        new_lambda.append(np.dot(theta_j, x_j) + agent.creator_arrival_rate(j+1))
    lambda_diff = np.array(new_lambda) - np.array(old_lambda)

    def negative_num(x):
        return x < 0

    lambda_diff = list(filter(negative_num, lambda_diff))  #只取小于0的部分
    diff_sum = np.sum(lambda_diff)

    reward = reward - 1000 * diff_sum
    return reward