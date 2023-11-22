import numpy as np
from conf import *
import copy
class Arm:
    def __init__(self,id=0,B=0.0,N=0.0,K=0.0,T=0,pb1=0.0,pb2=0.0,mean=0.0,cost=0.0,variance=0.0,reward_list=[]):
        self.id=id
        self.B = B
        self.N = N
        self.K = K
        self.pb1 = pb1  # privacy budget for rewards
        self.pb2 = pb2 # privacy budget for costs
        self.mean = mean # mean of the Laplace distribution
        self.cost = cost
        self.T = T
        self.variance = variance
        self.reward_list = reward_list # the original reward sequence released by the arm
        self.reward_size=len(reward_list)
        self.reward_list_backup= reward_list.copy()
        if self.reward_list == []:#
            self.reward_list = [i for i in np.random.normal(loc=self.mean, scale=self.variance, size=self.T)]
        self.reward_list_private = [] # the private reward sequence released by the arm
        self.cost_list=[i for i in np.random.normal(loc=self.cost, scale=self.variance, size=self.T)] # the original cost sequence released by the arm
        while min(self.cost_list)<=0:
            print('Arm creation <0 error. mean cost:%f'%(self.cost))
            self.cost_list=[i for i in np.random.normal(loc=self.cost, scale=self.variance, size=self.T)] # the original cost sequence released by the arm
        self.cost_list_private = [] # the private cost sequence released by the arm. The index is timeline.
        self.gamma_list=[0 for i in range(self.T)]# selection counter
        self.e_list=[0 for i in range(self.T)]# exploration index
        self.h_list=[0 for i in range(self.T)]# private index
        self.utility_list=[0 for i in range(self.T)]
        self.cost_max=max(self.cost_list)
        self.cost_min=min(self.cost_list)
        self.reward_max = max(self.reward_list)
        self.reward_min = min(self.reward_list)

    def obfuscate(self,delta_r,delta_c): # the obfuscation function for the first round
        self.reward_list_private=[]
        self.cost_list_private=[]
        temp_reward_list_private = self.reward_list + np.random.laplace(0, delta_r / self.pb1, len(self.reward_list))
        temp_cost_list_private = self.cost_list + np.random.laplace(0, delta_c / self.pb2, len(self.cost_list))
        for i in range(len(temp_reward_list_private)):  # calculate the privately released rewards
            self.reward_list_private.append(
                sum(temp_reward_list_private[:(i + 1)]) / len(temp_reward_list_private[:(i + 1)]))
        for i in range(len(temp_cost_list_private)):  # calculate the privately released costs
            self.cost_list_private.append(sum(temp_cost_list_private[:(i + 1)]) / len(temp_cost_list_private[:(i + 1)]))


    def __str__(self, t):
        return 'Arm (id,rcr): (' + str(self.id) + ',' + str(self.gamma_list[t]) + ')'

    def clear_changed_reward(self):
        self.reward_list = np.random.normal(loc=self.mean, scale=self.variance, size=self.T)
        # self.reward_list_private = []
        # self.cost_list_private = []
        self.utility_list = [0 for i in range(self.T)]
        self.gamma_list = [0 for i in range(self.T)]
        self.e_list = [0 for i in range(self.T)]
        self.h_list = [0 for i in range(self.T)]

    def clear_unchanged_reward(self):
        # self.reward_list_private = []
        # self.cost_list_private = []
        self.utility_list = [0 for i in range(self.T)]
        self.gamma_list = [0 for i in range(self.T)]
        self.e_list = [0 for i in range(self.T)]
        self.h_list = [0 for i in range(self.T)]


    def clear_unchanged_reward_Yahoo(self):
        # self.reward_list_private = []
        # self.cost_list_private = []
        self.reward_list = self.reward_list_backup.copy()
        self.utility_list = [0 for i in range(self.T)]
        self.gamma_list = [0 for i in range(self.T)]
        self.e_list = [0 for i in range(self.T)]
        self.h_list = [0 for i in range(self.T)]
