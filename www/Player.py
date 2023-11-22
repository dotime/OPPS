import math
import copy
import numpy as np
from www.conf import *
import time
from www.arm import Arm

class Player:
    def __init__(self):   #pb : privacy budget  T: current round
        self.B=Yahoo_B #budget
        self.N=Yahoo_N #arm number
        self.K=Yahoo_K #pulled arm number
        self.pb1=Yahoo_pb1 #privacy budget 1
        self.pb2=Yahoo_pb2 #privacy budget 2 for cost
        self.T=Yahoo_T #round number
        self.count=Yahoo_count # counter
        self.cost_mean = 10
        self.reward_mean = 20
        self.budget_list = [i for i in range(100, 1050, 50)] #ranges of several key variables
        self.K_list = [i / 10 for i in range(1, 6, 1)]
        self.variance_list = [i / 10 for i in range(1, 6, 1)]
        self.variance=1
        self.arm_list = []
        self.arm_mean_list=[]#the list consisting of all arms' means
        self.arm_cost_list=[]#the list consisting of all arms' costs
        self.cumulative_cost_list=[]
        self.cumulative_reward_list=[]
        self.mean_min=0.1
        self.mean_max=1
        self.cost_min=0.1
        self.cost_max=1
        self.reward_min=0
        self.reward_max=0
        self.final_t=0 #the last round's number. It differs in different policies given the same budget.

    def loadReward_Yahoo(self):
        dic = np.load(save_address + '\dic_T2.npy', allow_pickle=True).item()
        x = [len(dic[i]) for i in dic.keys()]
        y = [sum(dic[i]) for i in dic.keys()]
        alist=[i for i in dic.keys() if len(dic[i])/200>300]
        # print(len(alist))
        new_dic={}
        for i in alist:
            olist=dic[i]
            new_reward_list=[sum(olist[j:j+Article_step]) for j in range(0,len(olist),Article_step)]
            new_dic[i]=new_reward_list[:len(new_reward_list)-1] #didcard the last value in the list as it
        return new_dic

    def loadCost_Yahoo(self):
        dic = np.load(save_address + '\dic_cost.npy', allow_pickle=True).item()
        x = [len(dic[i]) for i in dic.keys()]
        y = [sum(dic[i]) for i in dic.keys()]
        alist=[i for i in dic.keys() if len(dic[i])/200>300]
        # print(len(alist))
        new_dic={}
        for i in alist:
            olist=dic[i]
            new_reward_list=[sum(olist[j:j+Article_step]) for j in range(0,len(olist),Article_step)]
            new_dic[i]=new_reward_list[:len(new_reward_list)-1] #didcard the last value in the list as it
        return new_dic

    #-------------------------------Initialization phase--------------------------------
    def create_arm_list(self):
        article_dic=self.loadReward_Yahoo()
        article_id_list=list(article_dic.keys())
        # article_cost_dic = np.load(save_address + '\dic_cost.npy', allow_pickle=True).item()
        self.arm_mean_list = [i for i in np.random.uniform(5, 15, size=self.N)]  # generate a mean list
        self.arm_cost_list = [i for i in np.random.uniform(5, 15, size=self.N)]  # generate a cost list
        self.arm_list = [Arm(id=k, B=self.B, N=self.N, K=self.K, T=self.T, pb1=self.pb1, pb2=self.pb2, mean=np.mean(article_dic[k]), cost=j,
                             variance=self.variance, reward_list=article_dic[k]) for (k,j) in zip(article_id_list,self.arm_cost_list)]
        self.arm_list=self.arm_list[:self.N]
        # if len(article_id_list)!=len(self.arm_list):
        #     print('error: wrong article number','N=',self.N,'K',self.K)
        #seek for the maximal and minimal values among rewards and costs
        temp_reward_list=[]
        temp_cost_list=[]
        for arm in self.arm_list:
            temp_reward_list=temp_reward_list+arm.reward_list
            temp_cost_list=temp_cost_list+arm.cost_list
        self.reward_min=min(temp_reward_list)
        self.reward_max=max(temp_reward_list)
        self.cost_max=max(temp_cost_list)
        self.cost_min=min(temp_cost_list)
        self.reward_max=15
        self.reward_min=5
        self.cost_min=5
        self.cost_max=15
        # print('r_max: %f, r_min: %f, c_max: %f, c_min: %f'%(self.reward_max,self.reward_min,self.cost_max,self.cost_min))
        for arm in self.arm_list:
            arm.reward_list_private=[0 for i in range(self.T)]
            arm.cost_list_private = [0 for i in range(self.T)]
        # print('Arm list created.')



    def clear_arm_list_unchanged_reward(self): #clear an arm list apart from the mean and cost
        for a in self.arm_list:
            a.clear_unchanged_reward()
    def clear_arm_list_changed_reward(self): #clear an arm list apart from the mean and cost
        for a in self.arm_list:
            a.clear_changed_reward()
    def clear_arm_list_changed_reward_Yahoo(self): #clear an arm list apart from the mean and cost
        for a in self.arm_list:
            a.clear_unchanged_reward_Yahoo()

    #-------------------------------BPP policy------------------------------------------
    def policy_bpp(self):
        # print('Bpp started.')
        if len(self.arm_list)==0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I=[] # collection of selected arms at time t
        t=0
        #Initialization Stage
        list_I.append(self.arm_list)
        for arm in list_I[t]:
            arm.gamma_list[t] = 1
            x = (arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            # temp_x = math.sqrt(math.log(t + 1) / arm.gamma_list[t])
            #e = arm.e_list[t] = (self.cost_min+self.reward_max) / (self.cost_min)**2 * temp_x
            e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
            #h = arm.h_list[t] = 4 / (self.cost_min) ** 2 * (
            #        arm.cost_min / self.pb1 + arm.reward_max / self.pb2) * temp_x
            h = arm.h_list[t] = (4 / (arm.cost_min) )* (
                    (arm.cost_min*(arm.reward_max-arm.reward_min)) / self.pb1 + (arm.reward_max*(arm.cost_max-arm.cost_min)) / self.pb2) * temp_x/ (arm.cost_min)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc + h + e
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        # self.cumulative_cost_list.append(sum([i.cost_list_private[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        if sum(self.cumulative_cost_list)>self.B:
            print('Error: an over-limit cumulative cost at round 1. sum:%f, B:%f'%(sum(self.cumulative_cost_list),self.B))
        #While loop until the budget is exhausted
        while sum(self.cumulative_cost_list)<=self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            #print('t:%d, budget expense: %f%%'%(t,sum(self.cumulative_cost_list) /self.B*100))
            t = t + 1
            self.arm_list=sorted(self.arm_list, key=lambda x:x.utility_list[t-1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm=self.arm_list[i]
                if (t >= len(arm.reward_list)):
                    print('Error: reward index out of range.', 't:', t, ' len(reward list):', len(arm.reward_list))
                if i<self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    while x<= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    for k in range(t,self.T):
                        arm.cost_list_private[k]=x
                        arm.reward_list_private[k]=y
                else:
                    len1=len(arm.reward_list)
                    arm.reward_list.insert(t,arm.reward_list[t]) # reserve the current reward for future use since it is not observed in this step
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increase the counter by 0 of the arms that are not selected in this step
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                # temp_x = math.sqrt( math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (arm.cost_min + arm.reward_max)/ (arm.cost_min)**2 * temp_x
                e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
                #e = arm.e_list[t] = (1 + self.reward_max / self.cost_min) * temp_x / (self.cost_min-temp_x)

                # h = arm.h_list[t] = (4 / (self.cost_min)) * (
                #         self.cost_min / self.pb1 + self.reward_max / self.pb2) * temp_x / (self.cost_min - temp_x)
                h = arm.h_list[t] = (4 / (arm.cost_min)) * (
                        (arm.cost_min * (arm.reward_max - arm.reward_min)) / self.pb1 + (
                            arm.reward_max * (arm.cost_max - arm.cost_min)) / self.pb2) * temp_x / (
                                                arm.cost_min)
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + h + e
                #print('BPP: h+e:%f , e: %f, x:%f, h:%f' % (h + e,  (self.cost_min + self.reward_max)/ (self.cost_min)**2 , temp_x, 4 / (self.cost_min) ** 2 * (
                #        self.cost_min / self.pb1 + self.reward_max / self.pb2) * temp_x))
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.dic_budget[self.B]=sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

    def policy_bpp_no_exploration_index(self):
        # print('Bpp started.')
        if len(self.arm_list)==0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_unchanged_reward()
        self.cumulative_reward_list=[]
        self.cumulative_cost_list=[]
        list_I=[] # collection of selected arms at time t
        t=0
        list_I.append(self.arm_list)
        for arm in list_I[t]:
            arm.gamma_list[t] = 1
            x = ( arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            # temp_x = math.sqrt(math.log(t + 1) / arm.gamma_list[t])
            #e = arm.e_list[t] = (self.cost_min+self.reward_max) / (self.cost_min)**2 * temp_x
            e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
            #h = arm.h_list[t] = 4 / (self.cost_min) ** 2 * (
            #        arm.cost_min / self.pb1 + arm.reward_max / self.pb2) * temp_x
            h = arm.h_list[t] = (4 / (arm.cost_min) )* (
                    (arm.cost_min*(arm.reward_max-arm.reward_min)) / self.pb1 + (arm.reward_max*(arm.cost_max-arm.cost_min)) / self.pb2) * temp_x/ (arm.cost_min)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc + h
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        # self.cumulative_cost_list.append(sum([i.cost_list_private[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        if sum(self.cumulative_cost_list)>self.B:
            print('Error: an over-limit cumulative cost at round 1. sum:%f, B:%f'%(sum(self.cumulative_cost_list),self.B))
        while sum(self.cumulative_cost_list)<=self.B:
            #print('t:%d, budget expense: %f%%'%(t,sum(self.cumulative_cost_list) /self.B*100))
            t = t + 1
            self.arm_list=sorted(self.arm_list, key=lambda x:x.utility_list[t-1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm=self.arm_list[i]
                if i<self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    while x<= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    for k in range(t,self.T):
                        arm.cost_list_private[k]=x
                        arm.reward_list_private[k]=y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                # temp_x = math.sqrt( math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (arm.cost_min + arm.reward_max)/ (arm.cost_min)**2 * temp_x
                e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
                #e = arm.e_list[t] = (1 + self.reward_max / self.cost_min) * temp_x / (self.cost_min-temp_x)

                # h = arm.h_list[t] = (4 / (self.cost_min)) * (
                #         self.cost_min / self.pb1 + self.reward_max / self.pb2) * temp_x / (self.cost_min - temp_x)
                h = arm.h_list[t] = (4 / (arm.cost_min)) * (
                        (arm.cost_min * (arm.reward_max - arm.reward_min)) / self.pb1 + (
                            arm.reward_max * (arm.cost_max - arm.cost_min)) / self.pb2) * temp_x / (
                                                arm.cost_min)
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + h
                #print('BPP: h+e:%f , e: %f, x:%f, h:%f' % (h + e,  (self.cost_min + self.reward_max)/ (self.cost_min)**2 , temp_x, 4 / (self.cost_min) ** 2 * (
                #        self.cost_min / self.pb1 + self.reward_max / self.pb2) * temp_x))
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t=t-1
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t])

    def policy_bpp_no_private_index(self):
        # print('Bpp started.')
        if len(self.arm_list)==0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_unchanged_reward()
        self.cumulative_reward_list=[]
        self.cumulative_cost_list=[]
        list_I=[] # collection of selected arms at time t
        t=0
        list_I.append(self.arm_list)
        for arm in list_I[t]:
            arm.gamma_list[t] = 1
            x = ( arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
            h = arm.h_list[t] = (4 / (arm.cost_min) )* (
                    (arm.cost_min*(arm.reward_max-arm.reward_min)) / self.pb1 + (arm.reward_max*(arm.cost_max-arm.cost_min)) / self.pb2) * temp_x/ (arm.cost_min)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc + e
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        if sum(self.cumulative_cost_list)>self.B:
            print('Error: an over-limit cumulative cost at round 1. sum:%f, B:%f'%(sum(self.cumulative_cost_list),self.B))
        while sum(self.cumulative_cost_list)<=self.B:
            t = t + 1
            self.arm_list=sorted(self.arm_list, key=lambda x:x.utility_list[t-1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm=self.arm_list[i]
                if i<self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    while x<= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    for k in range(t,self.T):
                        arm.cost_list_private[k]=x
                        arm.reward_list_private[k]=y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                # temp_x = math.sqrt( math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (arm.cost_min + arm.reward_max)/ (arm.cost_min)**2 * temp_x
                e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
                h = arm.h_list[t] = (4 / (arm.cost_min)) * (
                        (arm.cost_min * (arm.reward_max - arm.reward_min)) / self.pb1 + (
                            arm.reward_max * (arm.cost_max - arm.cost_min)) / self.pb2) * temp_x / (
                                                arm.cost_min)
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + e
                #print('BPP: h+e:%f , e: %f, x:%f, h:%f' % (h + e,  (self.cost_min + self.reward_max)/ (self.cost_min)**2 , temp_x, 4 / (self.cost_min) ** 2 * (
                #        self.cost_min / self.pb1 + self.reward_max / self.pb2) * temp_x))
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t=t-1
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t])

    def policy_bpp_T(self): #with the T constarint rather budget constarint.
        # print('Bpp started.')
        if len(self.arm_list)==0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_unchanged_reward()
        self.cumulative_reward_list=[]
        self.cumulative_cost_list=[]
        list_I=[] # collection of selected arms at time t
        t=0
        list_I.append(self.arm_list)
        for arm in list_I[t]:
            arm.gamma_list[t] = 1
            x = ( arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            # temp_x = math.sqrt(math.log(t + 1) / arm.gamma_list[t])
            #e = arm.e_list[t] = (self.cost_min+self.reward_max) / (self.cost_min)**2 * temp_x
            e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
            #h = arm.h_list[t] = 4 / (self.cost_min) ** 2 * (
            #        arm.cost_min / self.pb1 + arm.reward_max / self.pb2) * temp_x
            h = arm.h_list[t] = (4 / (arm.cost_min) )* (
                    (arm.cost_min*(arm.reward_max-arm.reward_min)) / self.pb1 + (arm.reward_max*(arm.cost_max-arm.cost_min)) / self.pb2) * temp_x/ (arm.cost_min)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc + h + e
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        # self.cumulative_cost_list.append(sum([i.cost_list_private[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        if sum(self.cumulative_cost_list)>self.B:
            print('Error: an over-limit cumulative cost at round 1. sum:%f, B:%f'%(sum(self.cumulative_cost_list),self.B))
        while sum(self.cumulative_cost_list)<=self.B:
            #print('t:%d, budget expense: %f%%'%(t,sum(self.cumulative_cost_list) /self.B*100))
            t = t + 1
            self.arm_list=sorted(self.arm_list, key=lambda x:x.utility_list[t-1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm=self.arm_list[i]
                if i<self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    while x<= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    for k in range(t,self.T):
                        arm.cost_list_private[k]=x
                        arm.reward_list_private[k]=y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                # temp_x = math.sqrt( math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (arm.cost_min + arm.reward_max)/ (arm.cost_min)**2 * temp_x
                e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
                #e = arm.e_list[t] = (1 + self.reward_max / self.cost_min) * temp_x / (self.cost_min-temp_x)

                # h = arm.h_list[t] = (4 / (self.cost_min)) * (
                #         self.cost_min / self.pb1 + self.reward_max / self.pb2) * temp_x / (self.cost_min - temp_x)
                h = arm.h_list[t] = (4 / (arm.cost_min)) * (
                        (arm.cost_min * (arm.reward_max - arm.reward_min)) / self.pb1 + (
                            arm.reward_max * (arm.cost_max - arm.cost_min)) / self.pb2) * temp_x / (
                                                arm.cost_min)
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + h + e
                #print('BPP: h+e:%f , e: %f, x:%f, h:%f' % (h + e,  (self.cost_min + self.reward_max)/ (self.cost_min)**2 , temp_x, 4 / (self.cost_min) ** 2 * (
                #        self.cost_min / self.pb1 + self.reward_max / self.pb2) * temp_x))
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t=t-1
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t])

    # metrics: cumulative reward, regret, privacy leakage, running time.
    #arguments: K,B,pb
    # -------------------------------optiaml policy----------------------------------------
    def policy_optimal(self):
        # print('Optimal started.')
        if len(self.arm_list)==0:
            print('Error: Empty arm list.')
            return -1,-1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_cost_list=[]
        self.cumulative_reward_list=[]
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        # self.arm_list = sorted(self.arm_list, key=lambda x: x.reward_list[t] / x.cost_list[t], reverse=True)
        self.arm_list = sorted(self.arm_list, key=lambda x: x.mean / x.cost, reverse=True)
        temp_list_I = self.arm_list[:self.K]
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            # print('t:%d, budget expense: %f%%' % (t, sum(self.cumulative_cost_list)  / self.B*100))
            t = t + 1
            # self.arm_list = sorted(self.arm_list, key=lambda x: x.reward_list[t] / x.cost_list[t], reverse=True)
            # list_I.append(self.arm_list[:self.K])
            list_I.append(temp_list_I)
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i >= self.K:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

    # -------------------------------no privacy policy----------------------------------------
    def policy_no_privacy(self):
        # print('UCB-MB no privacy started.')
        if len(self.arm_list) == 0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        for arm in self.arm_list:
            arm.gamma_list[t] = 1
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            x = arm.cost_list[t]  / arm.gamma_list[t]
            y = arm.reward_list[t] /  arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min - temp_x)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc + e
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            t = t + 1
            self.arm_list = sorted(self.arm_list, key=lambda x: x.utility_list[t - 1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i < self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] ) / \
                        arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t]) / \
                        arm.gamma_list[t]
                    for k in range(t, self.T):
                        arm.cost_list_private[k] = x
                        arm.reward_list_private[k] = y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min - temp_x)
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + e
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t = t - 1
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

    def policy_no_privacy_aucb(self):
        # print('AUCB no privacy started.')
        if len(self.arm_list) == 0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        for arm in self.arm_list:
            arm.gamma_list[t] = 1
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            x = arm.cost_list[t] / arm.gamma_list[t]
            y = arm.reward_list[t] / arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            e = arm.e_list[t] = (1 + self.reward_max / self.cost_min) * temp_x / (self.cost_min - temp_x)
            rc = arm.reward_list_private[t] / arm.cost_list[t]
            u = rc + e
            e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc + e
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            # print('t:%d, budget expense: %f%%' % (t, sum(self.cumulative_cost_list) / self.B * 100))
            t = t + 1
            self.arm_list = sorted(self.arm_list, key=lambda x: x.utility_list[t - 1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i < self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t]) / \
                        arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t]) / \
                        arm.gamma_list[t]
                    for k in range(t, self.T):
                        arm.cost_list_private[k] = x
                        arm.reward_list_private[k] = y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (1 + self.reward_max / self.cost_min) * temp_x / (self.cost_min - temp_x)
                rc = arm.reward_list_private[t] / arm.cost_list[t]
                u = rc + temp_x/arm.cost_list[t]
                e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min )
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + e
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t = t - 1
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

    # -------------------------------UCB-MB----------------------------------------
    def policy_ucb_mb(self):
        # print('UCB-MB started.')
        if len(self.arm_list) == 0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        for arm in self.arm_list:
            arm.gamma_list[t] = 1
            x = (arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            e = arm.e_list[t] = (1+arm.reward_max/arm.cost_min)*temp_x/(arm.cost_min-temp_x)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc +  e
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            #print('t:%d, budget expense: %f%%' % (t, sum(self.cumulative_cost_list) / self.B * 100))
            t = t + 1
            self.arm_list = sorted(self.arm_list, key=lambda x: x.utility_list[t - 1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i < self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                        arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                        arm.gamma_list[t]
                    while x <= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                            arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[
                                 0]) / arm.gamma_list[t]
                    for k in range(t, self.T):
                        arm.cost_list_private[k] = x
                        arm.reward_list_private[k] = y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (1 + arm.reward_max/ arm.cost_min) * temp_x / (arm.cost_min - temp_x)
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + e
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t = t - 1
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

    # -------------------------------CUCB-LDP1----------------------------------------

    # -------------------------------AUCB ---------------------------------------

    def policy_aucb(self):
        # print('aucb started.')
        if len(self.arm_list) == 0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        for arm in self.arm_list:
            arm.gamma_list[t] = 1
            x = (arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            e = arm.e_list[t] = temp_x
            rc = (arm.reward_list_private[t] + temp_x) / arm.cost_list_private[t]
            u = rc
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            #print('t:%d, budget expense: %f%%' % (t, sum(self.cumulative_cost_list) / self.B * 100))
            t = t + 1
            self.arm_list = sorted(self.arm_list, key=lambda x: x.utility_list[t - 1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i < self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                        arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                        arm.gamma_list[t]
                    while x <= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                            arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[
                                 0]) / arm.gamma_list[t]
                    for k in range(t, self.T):
                        arm.cost_list_private[k] = x
                        arm.reward_list_private[k] = y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = temp_x
                rc = (arm.reward_list_private[t]+temp_x) / arm.cost_list_private[t]
                u = rc
                #print('aucb: h+e:%f, x:%f' % (e/arm.cost_list_private[0],temp_x))
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t = t - 1
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget





    def policy_exploration(self):
        # print('exploration started.')
        if len(self.arm_list) == 0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        for arm in self.arm_list:
            arm.gamma_list[t] = 1
            x = (arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            e = arm.e_list[t] = temp_x
            rc = (arm.reward_list_private[t] + temp_x) / arm.cost_list_private[t]
            u = (arm.reward_list_private[t]) / arm.cost_list_private[t]
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            # print('t:%d, budget expense: %f%%' % (t, sum(self.cumulative_cost_list) / self.B * 100))
            t = t + 1
            self.arm_list = sorted(self.arm_list, key=lambda x: x.gamma_list[t - 1], reverse=False)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i < self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                        arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                        arm.gamma_list[t]
                    while x <= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                            arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[
                                 0]) / arm.gamma_list[t]
                    for k in range(t, self.T):
                        arm.cost_list_private[k] = x
                        arm.reward_list_private[k] = y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = temp_x
                rc = (arm.reward_list_private[t] + temp_x) / arm.cost_list_private[t]
                u = rc
                u = (arm.reward_list_private[t]) / arm.cost_list_private[t]
                # print('aucb: h+e:%f, x:%f' % (e / arm.cost_list_private[0], temp_x))
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t = t - 1
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

    def policy_exploitation(self):
        # print('exploitation started.')
        if len(self.arm_list) == 0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        for arm in self.arm_list:
            arm.gamma_list[t] = 1
            x = (arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            e = arm.e_list[t] = temp_x
            rc = (arm.reward_list_private[t] + temp_x) / arm.cost_list_private[t]
            u =  (arm.reward_list_private[t]) / arm.cost_list_private[t]
            arm.utility_list[t] = u
        #self.cumulative_cost_list.append(sum([i.cost_list_private[t] for i in list_I[t]]))
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            #print('t:%d, budget expense: %f%%' % (t, sum(self.cumulative_cost_list) / self.B * 100))
            t = t + 1
            self.arm_list = sorted(self.arm_list, key=lambda x: x.utility_list[t - 1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i < self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                        arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                        arm.gamma_list[t]
                    while x <= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                            arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[
                                 0]) / arm.gamma_list[t]
                    for k in range(t, self.T):
                        arm.cost_list_private[k] = x
                        arm.reward_list_private[k] = y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = temp_x
                rc = (arm.reward_list_private[t] + temp_x) / arm.cost_list_private[t]
                u = rc
                u = arm.reward_list_private[t]/ arm.cost_list_private[t]
                #print('aucb: h+e:%f, x:%f' % (e / arm.cost_list_private[0], temp_x))
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t = t - 1
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

    def policy_CUCB_DP(self):
        # print('cucb-dp started.')
        if len(self.arm_list) == 0:
            print('Error: Empty arm list.')
            return -1, -1
        self.clear_arm_list_changed_reward_Yahoo()
        self.cumulative_reward_list = []
        self.cumulative_cost_list = []
        self.dic_budget={}
        list_I = []  # collection of selected arms at time t
        t = 0
        list_I.append(self.arm_list)
        for arm in self.arm_list:
            arm.gamma_list[t] = 1
            x = (arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            eT=1+(self.B-self.N*(self.cost_min+self.cost_max)/2)/(self.K*(self.cost_min+self.cost_max)/2)
            try:
                e = 12 * self.K * math.log(eT) ** 3 / (arm.gamma_list[t] * self.pb1)
            except ValueError:
                print(eT)
            temp_x = math.sqrt(4* math.log(self.N*eT) / arm.gamma_list[t])
            rc = (arm.reward_list_private[t])
            u =  min(rc+temp_x+e,1)/ arm.cost_list_private[t]
            arm.utility_list[t] = u
        #self.cumulative_cost_list.append(sum([i.cost_list_private[t] for i in list_I[t]]))
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        while sum(self.cumulative_cost_list) <= self.B:
            if sum(self.cumulative_cost_list)//100>sum(self.cumulative_cost_list[:-1])//100:
                self.dic_budget[sum(self.cumulative_cost_list)//100*100]=sum(self.cumulative_reward_list[:-1])
            #print('t:%d, budget expense: %f%%' % (t, sum(self.cumulative_cost_list) / self.B * 100))
            t = t + 1
            self.arm_list = sorted(self.arm_list, key=lambda x: x.utility_list[t - 1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm = self.arm_list[i]
                if i < self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                        arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[0]) / \
                        arm.gamma_list[t]
                    while x <= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(self.cost_max - self.cost_min) / self.pb2, size=1)[0]) / \
                            arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(self.reward_max - self.reward_min) / self.pb1, size=1)[
                                 0]) / arm.gamma_list[t]
                    for k in range(t, self.T):
                        arm.cost_list_private[k] = x
                        arm.reward_list_private[k] = y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                eT = 1 + (self.B - self.N * (self.cost_min + self.cost_max) / 2) / (
                            self.K * (self.cost_min + self.cost_max) / 2)
                e = 12 * self.K * math.log(eT) ** 3 / (arm.gamma_list[t] * self.pb1)
                temp_x = math.sqrt(4 * math.log(self.N * eT) / arm.gamma_list[t])
                rc = (arm.reward_list_private[t])
                u = min(rc + temp_x + e, 1) / arm.cost_list_private[t]
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t = t - 1
        self.dic_budget[self.B] = sum(self.cumulative_reward_list[:-1])
        return sum(self.cumulative_reward_list[:t]), sum(self.cumulative_cost_list[:t]), t, self.dic_budget

#---------------------Large-scale simulations----------------------
    def policy_payment(self,delta):
        self.clear_arm_list_unchanged_reward()
        self.cumulative_reward_list=[]
        self.cumulative_cost_list=[]
        list_I=[] # collection of selected arms at time t
        t=0
        list_I.append(self.arm_list)
        for arm in list_I[t]:
            arm.gamma_list[t] = 1
            x = ( arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                arm.gamma_list[t]
            y = (arm.reward_list[t] + \
                 np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                arm.gamma_list[t]
            while x <= 0:
                x = (arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / \
                    arm.gamma_list[t]
            while y <= 0:
                y = (arm.reward_list[t] + \
                     np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / \
                    arm.gamma_list[t]
            for k in range(t, self.T):
                arm.cost_list_private[k] = x
                arm.reward_list_private[k] = y
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
            e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
            h = arm.h_list[t] = (4 / (arm.cost_min) )* (
                    (arm.cost_min*(arm.reward_max-arm.reward_min)) / self.pb1 + (arm.reward_max*(arm.cost_max-arm.cost_min)) / self.pb2) * temp_x/ (arm.cost_min)
            rc = arm.reward_list_private[t] / arm.cost_list_private[t]
            u = rc + h + e
            arm.utility_list[t] = u
        self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
        self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        if sum(self.cumulative_cost_list)>self.B:
            print('Error: an over-limit cumulative cost at round 1. sum:%f, B:%f'%(sum(self.cumulative_cost_list),self.B))
        while sum(self.cumulative_cost_list)<=self.B:
            t = t + 1
            self.arm_list=sorted(self.arm_list, key=lambda x:x.utility_list[t-1], reverse=True)
            list_I.append(self.arm_list[:self.K])
            for i in range(len(self.arm_list)):
                arm=self.arm_list[i]
                if i<self.K:
                    arm.gamma_list[t] = arm.gamma_list[t - 1] + 1  # increased the counter of selected arms by 1
                    x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                         np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    while x<= 0:
                        x = (arm.cost_list_private[t - 1] * arm.gamma_list[t - 1] + arm.cost_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.cost_max - arm.cost_min) / self.pb2, size=1)[0]) / arm.gamma_list[t]
                    while y <= 0:
                        y = (arm.reward_list_private[t - 1] * arm.gamma_list[t - 1] + arm.reward_list[t] + \
                             np.random.laplace(loc=0, scale=(arm.reward_max - arm.reward_min) / self.pb1, size=1)[0]) / arm.gamma_list[t]
                    for k in range(t,self.T):
                        arm.cost_list_private[k]=x
                        arm.reward_list_private[k]=y
                else:
                    arm.reward_list.insert(t, arm.reward_list[t])  # reserve the current reward for future use
                    arm.gamma_list[t] = arm.gamma_list[t - 1]  # increased the counter of not selected arms by 0
                temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / arm.gamma_list[t])
                e = arm.e_list[t] = (1 + arm.reward_max / arm.cost_min) * temp_x / (arm.cost_min)
                h = arm.h_list[t] = (4 / (arm.cost_min)) * (
                        (arm.cost_min * (arm.reward_max - arm.reward_min)) / self.pb1 + (
                            arm.reward_max * (arm.cost_max - arm.cost_min)) / self.pb2) * temp_x / (
                                                arm.cost_min)
                rc = arm.reward_list_private[t] / arm.cost_list_private[t]
                u = rc + h + e
                arm.utility_list[t] = u
            self.cumulative_cost_list.append(sum([i.cost_list[t] for i in list_I[t]]))
            self.cumulative_reward_list.append(sum([i.reward_list[t] for i in list_I[t]]))
        self.final_t=t-1

        p1,e1,p2,e2,p_max=self.mean_gain_winner(armlist=self.arm_list,t=self.final_t,delta=delta)
        return p1,e1,p2,e2,p_max


    def mean_gain_winner(self, armlist,t,delta): #average profit gained by lying
        temp_list=copy.deepcopy(armlist)
        #calculate the expected payment when truthful.
        temp_arm_id=0
        temp_arm=None
        max1=0
        truthful_payment = []
        for arm in temp_list[self.K:]:
        #for arm in temp_list[:self.K]:
            if arm.gamma_list[t]>max1:
                max1=arm.gamma_list[t]
                temp_arm_id=arm.id
                temp_arm=arm
        for i in range(500):
            x = (temp_arm.cost_list_private[t - 1] * temp_arm.gamma_list[t - 1] + temp_arm.cost_list[t] + \
                 np.random.laplace(loc=0, scale=(temp_arm.cost_max - temp_arm.cost_min) / self.pb2, size=1)[0]) / temp_arm.gamma_list[
                    t]
            # print('gamma:',temp_arm.gamma_list[t - 1],'gamma+1:',temp_arm.gamma_list[t])
            while x <= 0:
                x = (temp_arm.cost_list_private[t - 1] * temp_arm.gamma_list[t - 1] + temp_arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(temp_arm.cost_max - temp_arm.cost_min) / self.pb2, size=1)[0]) / \
                    temp_arm.gamma_list[
                        t]
            for k in range(t, self.T):
                temp_arm.cost_list_private[k] = x
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / temp_arm.gamma_list[t])
            e = temp_arm.e_list[t] = (1 + temp_arm.reward_max / temp_arm.cost_min) * temp_x / (temp_arm.cost_min)
            h = temp_arm.h_list[t] = (4 / (temp_arm.cost_min)) * (
                    (temp_arm.cost_min * (temp_arm.reward_max - temp_arm.reward_min)) / self.pb1 + (
                        temp_arm.reward_max * (temp_arm.cost_max - temp_arm.cost_min)) / self.pb2) * temp_x / (temp_arm.cost_min)
            rc = temp_arm.reward_list_private[t] / temp_arm.cost_list_private[t]
            u = rc + h + e
            temp_arm.utility_list[t] = u
            temp_list = sorted(temp_list, key=lambda x: x.utility_list[t], reverse=True)
            index=temp_list.index(temp_arm)
            if index<self.K:
                a=temp_list[index+1]
                payment=temp_arm.utility_list[t]/(a.utility_list[t]/a.cost_list_private[t]+(a.e_list[t]+a.h_list[t])-(temp_arm.e_list[t]+temp_arm.h_list[t]))
                if payment<0:
                    truthful_payment.append(temp_arm.cost_max)
                else:
                    truthful_payment.append(payment)
                # print(temp_arm.utility_list[t]>a.utility_list[t],'index:',index,'truthful_payment:',payment,'cost_private:',temp_arm.cost_list_private[t] ,'cost:',temp_arm.cost_list[t] )
            else:
                truthful_payment.append(0)

        untruthful_payment = []
        for i in range(500):
            x = (temp_arm.cost_list_private[t - 1] * temp_arm.gamma_list[t - 1] + temp_arm.cost_list[t]  + \
                 np.random.laplace(loc=0, scale=(temp_arm.cost_max - temp_arm.cost_min) / self.pb2, size=1)[0]) / temp_arm.gamma_list[
                    t]+ delta
            while x <= 0:
                x = (temp_arm.cost_list_private[t - 1] * temp_arm.gamma_list[t - 1] + temp_arm.cost_list[t] + \
                     np.random.laplace(loc=0, scale=(temp_arm.cost_max - temp_arm.cost_min) / self.pb2, size=1)[0]) / \
                    temp_arm.gamma_list[
                        t]+ delta
            for k in range(t, self.T):
                temp_arm.cost_list_private[k] = x
            temp_x = math.sqrt((self.K + 1) * math.log(t + 1) / temp_arm.gamma_list[t])
            e = temp_arm.e_list[t] = (1 + temp_arm.reward_max / temp_arm.cost_min) * temp_x / (temp_arm.cost_min)
            h = temp_arm.h_list[t] = (4 / (temp_arm.cost_min)) * (
                    (temp_arm.cost_min * (temp_arm.reward_max - temp_arm.reward_min)) / self.pb1 + (
                        temp_arm.reward_max * (temp_arm.cost_max - temp_arm.cost_min)) / self.pb2) * temp_x / (temp_arm.cost_min)
            rc = temp_arm.reward_list_private[t] / temp_arm.cost_list_private[t]
            u = rc + h + e
            temp_arm.utility_list[t] = u
            temp_list = sorted(temp_list, key=lambda x: x.utility_list[t], reverse=True)
            index=temp_list.index(temp_arm)
            if index<self.K:
                a=temp_list[index+1]
                payment=temp_arm.utility_list[t]/(a.utility_list[t]/a.cost_list_private[t]+(a.e_list[t]+a.h_list[t])-(temp_arm.e_list[t]+temp_arm.h_list[t]))
                if payment < 0:
                    untruthful_payment.append(temp_arm.cost_max)
                else:
                    untruthful_payment.append(payment)
                # print('index:', index, 'untruthful_payment:', payment, 'cost_private:', temp_arm.cost_list_private[t],
                #       'cost:', temp_arm.cost_list[t])
            else:
                untruthful_payment.append(0)
        return np.mean(truthful_payment), np.std(truthful_payment), np.mean(untruthful_payment), np.std(untruthful_payment), max(truthful_payment+untruthful_payment)

