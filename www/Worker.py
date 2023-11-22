#Class for the platform, workers, tasks
import numpy as np
import math
import copy

class Worker:
    def __init__(self, id=-1, B=-1, T=-1, n=-1,mean=-1,cost=-1,variance=-1, bundle=[], x=-1,y=-1, capacity=-1,payment=-1):
        self.id=id
        self.B=B
        self.n=n
        self.T=T
        self.capacity=capacity
        self.mean=mean
        self.cost=cost
        self.variance=variance
        self.bundle=bundle
        self.e_list = [0 for i in range(self.T)]
        self.x=x
        self.y=y
        self.task_sequence = []
        self.payment=payment
        self.other_payment=0
        self.reward_list=[]

    def calUtility(self):
        self.offline_utility=len(self.bundle)
        self.mean=len(self.bundle)
        self.online_utility_list = [i for i in np.random.normal(loc=self.mean, scale=self.variance, size=self.T+1)]
        self.online_utility_list_copy=copy.deepcopy(self.online_utility_list)
        self.reward_list.append(self.online_utility_list[0])  #each worker is assigned at least once

    def __str__(self):
        return 'Worker (id,{task}):'+'({},{})'.format(self.id,self.bundle)

    def recover_utility(self):
        self.online_utility_list = copy.deepcopy(self.online_utility_list_copy)

class Task:
    def __init__(self, id=-1, x=-1, y=-1, worker_id=-1):
        self.id=id
        self.x=x
        self.y=y
        self.worker_id=worker_id

if __name__ == '__main__':
    print("Define worker class and task class.")
