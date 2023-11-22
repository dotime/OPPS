import math
import copy
import numpy as np
import pandas as pd
from  conf import *
import random
from Worker import *
import itertools
import time

class Policy:
    def __init__(self):
        self.T = T
        self.n = n
        self.m = m
        self.B = B
        self.size = size
        self.count = count
        self.variance = variance

        self.worker_list=[]
        self.task_list=[]
        self.task_bundle_list=[]

    # -------------------------------several assistant functions-----------------------------------
    def reset(self):
        self.T = T
        self.n = n
        self.m = m
        self.B = B
        self.size = size
        self.count = count
        self.variance = variance

    def calDistance(self,i,j):
        return math.sqrt((i.x-j.x)**2+(i.y-j.y)**2)

    def calCost(self, worker,task_list):
        task_sequence=[]
        temp_list=copy.deepcopy(task_list)
        sum=0
        min = 100000000000
        index = -1
        for j in range(len(temp_list)):
            task = temp_list[j]
            temp = self.calDistance(worker, task)
            if temp <= min:
                min = temp
                index = j
        temp_task=temp_list[index]
        task_sequence.append(temp_list[index])
        temp_list.remove(temp_list[index])
        sum = sum + min
        while(len(temp_list)>0):
            min=100000000000
            index=-1
            for j in range(len(temp_list)):
                task=temp_list[j]
                temp=self.calDistance(temp_task,task)
                if temp<=min:
                    min=temp
                    index=j
            temp_task=temp_list[index]
            task_sequence.append(temp_list[index])
            temp_list.remove(temp_list[index])
            sum=sum+min
        return sum, task_sequence

    '''Calculate the conflicting count among workers'''
    def ifconflict_1(self, worker_list):
        count=0
        for i in range(len(worker_list)):
            w1=worker_list[i]
            for j in range(i+1,len(worker_list)):
                w2=worker_list[j]
                x=[i for i in w1.bundle if i in w2.bundle]
                if len(x)>0:
                    count=count+1
        return count

    def ifconflict_2(self,worker,worker_list):
        count=0
        for temp_worker in worker_list:
            x = [i for i in worker.bundle if i in temp_worker.bundle]
            if len(x) > 0:
                count = count + 1
        return count

    def calUtility(self, list_list):
        utility=[]
        for list in list_list:
            utility.append(sum(i.offline_utility for i in list))
        return utility

    def calBudget(self):
        worker_list, task_list = self.loadWorkers(city='Tokyo')
        B_list=range(50,901,50)
        d={}
        for b in B_list:
            u3 = self.off_policy_TRAC_2(worker_list=worker_list, budget=b)
            d[b]='{},{}'.format(len(u3),len(worker_list))
        return d

    def caln(self):
        n_list=range(40,101,10)
        d={}
        for x in n_list:
            self.n=x
            worker_list, task_list = self.loadWorkers(city='Tokyo')
            u3 = self.off_policy_TRAC_2(worker_list=worker_list, budget=self.B)
            d[x]='winner size:{} worker size:{} budget used:{} utility:{}'.\
                format(len(u3),len(worker_list),sum([i.cost for i in u3]), sum([i.offline_utility for i in u3]))
        return d


#-------------------------------initialization functon-----------------------------------
    '''load workers from datasets'''
    def loadWorkers(self, city):
        worker_list = []
        task_list = []
        df = None
        if city == 'Tokyo':
            df = pd.read_excel(dataset_address + '\dataset_TSMC2014_TKY.xlsx')
        if city == 'New York':
            df = pd.read_excel(dataset_address + '\dataset_TSMC2014_NYC.xlsx')
        df = df.drop_duplicates(['latitude', 'longitude'])  # 用来去除重复值，防止出现distance=0的情况
        ser = df['userid']
        ser = ser.drop_duplicates()
        list1 = ser.to_list()
        groupdata = df.groupby('userid')
        count = 0
        for i in list1:
            temp = groupdata.get_group(i)
            lat = temp.iloc[0, 1]  # 取的是第一个用户
            longt = temp.iloc[0, 2]
            time = temp.iloc[0, 5]
            cap = temp.iloc[0, 4]
            # cap = capacity
            if count < self.n:
                worker = Worker(x=longt, y=lat, id=i, capacity=cap, B=self.B, T=self.T, n=self.n, variance=self.variance)
                worker_list.append(worker)
            elif count < self.n + self.m:
                task = Task(id=i, x=longt, y=lat)
                task_list.append(task)
            elif count >= self.n + self.m:
                break
            count = count + 1
        # after tasks are identifyed, we then decide each worker's task bundle and correponding cost
        for i in worker_list:
            i.bundle = random.sample(task_list, i.capacity)
            i.calUtility()
            i.cost, i.task_sequence = self.calCost(i, i.bundle)
            for j in i.bundle:
                j.worker_id = i.id
        # print('Conflict count:{}'.format(self.ifconflict_1(worker_list)))
        return worker_list, task_list

    # -------------------------------offline policies-----------------------------------
    def off_policy_CAP(self, worker_list, budget=B):
        sum_cost=0
        sum_utility=0
        temp_worker_list=copy.deepcopy(worker_list)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.offline_utility/math.sqrt(x.cost), reverse=True)
        winner_list=[]
        while(budget>sum_cost):
            if(len(temp_worker_list)==0):
                print('------Error: Too much Budget and too few workers')
            worker=temp_worker_list.pop(0)
            if self.ifconflict_2(worker=worker,worker_list=winner_list)==0:
                winner_list.append(worker)
                sum_cost=sum_cost+worker.cost
                sum_utility=sum_utility+worker.offline_utility
            # else:
            #     print('worker {} conflicted with winners.'.format(worker.id))
        winner_list=winner_list[:-1]
        return winner_list

    def off_policy_Optimal(self, worker_list, budget=B):
        sum_utility=0
        winner_set=[]
        temp_worker_list=copy.deepcopy(worker_list)
        for x in range(len(temp_worker_list)):
            for temp_list in itertools.combinations(temp_worker_list,x+1):
                if sum([i.cost for i in temp_list])<budget:
                    temp=sum([i.offline_utility for i in temp_list])
                    if temp>sum_utility:
                        sum_utility = temp
                        winner_set = temp_list
        return winner_set, sum_utility

    def off_policy_CPP(self, worker_list, budget=B):
        sum_cost=0
        sum_utility=0
        temp_worker_list=copy.deepcopy(worker_list)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.offline_utility/math.sqrt(x.cost), reverse=True)
        winner_list=[]
        while(budget>sum_cost):
            if(len(temp_worker_list)==0):
                print('------Error: Too much Budget and too few workers')
            worker=temp_worker_list.pop(0)
            if self.ifconflict_2(worker=worker,worker_list=winner_list)==0:# no conflict
                winner_list.append(worker)
                sum_cost=sum_cost+worker.cost
                sum_utility=sum_utility+worker.offline_utility
            else:
                for w in winner_list:
                    if len([i for i in w.bundle if i in worker.bundle])>0:
                        w.payment=(w.offline_utility/worker.offline_utility)**2*worker.cost
                        w.other_payment=w.payment
                        # print("*******conflict******", w.payment, w.cost)
        kk=winner_list[-1]
        winner_list=winner_list[:-1]

        '''pay each winner'''
        temp_worker_list = copy.deepcopy(worker_list)
        indicator_list=[]
        for w in winner_list:
            if w.payment>0:
                continue
            list_without_w=[i for i in temp_worker_list if i.id!=w.id]
            winner_without_w=self.off_policy_CAP(worker_list=list_without_w,budget=self.B)
            winner_without_w=sorted(winner_without_w, key=lambda x: x.offline_utility/math.sqrt(x.cost), reverse=True)
            if w.payment<=0:
                k = winner_without_w.pop(-1)
                k=kk
                w.payment = (w.offline_utility / k.offline_utility) ** 2 * k.cost
                w.other_payment = 0
                # print('----2--A case occurs where payment<0---------',len(winner_without_w),len(winner_list))
                # while(True):
                #     k=winner_without_w.pop(-1)
                #     if sum([i.cost for i in winner_without_w if i.id!=k.id])+w.cost<=self.B:
                #         w.payment = (w.offline_utility / k.offline_utility) ** 2 * k.cost
                #         w.other_payment=0
                #         indicator_list.append(1)
                #         break
        return winner_list

    def off_policy_random(self,worker_list,budget=B):
        sum_cost = 0
        sum_utility = 0
        temp_worker_list = copy.deepcopy(worker_list)
        winner_list = []
        while (budget > sum_cost):
            random_index=random.randint(0, len(temp_worker_list)-1)
            worker = temp_worker_list.pop(random_index)
            if self.ifconflict_2(worker=worker, worker_list=winner_list) == 0:
                winner_list.append(worker)
                sum_cost = sum_cost + worker.cost
                sum_utility = sum_utility + worker.offline_utility
            # else:
                # print('worker {} conflicted with winners.'.format(worker.id))
        winner_list = winner_list[:-1]
        return winner_list

    def off_policy_TRAC_2(self, worker_list, budget=B):
        sum_cost = 0
        sum_utility = 0
        temp_worker_list = copy.deepcopy(worker_list)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.offline_utility / x.cost, reverse=True)
        winner_list = []
        while (budget > sum_cost):
            worker = temp_worker_list.pop(0)
            if self.ifconflict_2(worker=worker, worker_list=winner_list) == 0:
                winner_list.append(worker)
                sum_cost = sum_cost + worker.cost
                sum_utility = sum_utility + worker.offline_utility
            else:
                if self.size>1:
                    sum_cost = sum_cost + worker.cost  # uneffective worker
        winner_list = winner_list[:-1]
        return winner_list

    def off_policy_TRAC(self, worker_list, budget=B):
        sum_cost = 0
        sum_utility = 0
        temp_worker_list = copy.deepcopy(worker_list)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.offline_utility / x.cost, reverse=True)
        winner_list = []
        while (budget > sum_cost):
            worker = temp_worker_list.pop(0)
            if self.ifconflict_2(worker=worker, worker_list=winner_list) == 0:
                winner_list.append(worker)
                sum_cost = sum_cost + worker.cost
                sum_utility = sum_utility + worker.offline_utility
            # else:
                # print('worker {} conflicted with winners.'.format(worker.id))
        winner_list = winner_list[:-1]
        return winner_list

    def off_policy_greedy(self, worker_list, budget=B):
        sum_cost = 0
        sum_utility = 0
        temp_worker_list = copy.deepcopy(worker_list)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.offline_utility, reverse=True)
        winner_list = []
        while (budget > sum_cost):
            worker = temp_worker_list.pop(0)
            if self.ifconflict_2(worker=worker, worker_list=winner_list) == 0:
                winner_list.append(worker)
                sum_cost = sum_cost + worker.cost
                sum_utility = sum_utility + worker.offline_utility
            # else:
                # print('worker {} conflicted with winners.'.format(worker.id))
        winner_list = winner_list[:-1]
        return winner_list

    def off_policy_K(self, worker_list, budget=B, K=-1):
        sum_cost = 0
        sum_utility = 0
        temp_worker_list = copy.deepcopy(worker_list)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.offline_utility / x.cost, reverse=True)
        winner_list = []
        # K=self.calK(worker_list=temp_worker_list,budget=budget)
        while (K > len(winner_list)):
            worker = temp_worker_list.pop(0)
            if self.ifconflict_2(worker=worker, worker_list=winner_list) == 0:
                winner_list.append(worker)
                sum_cost = sum_cost + worker.cost
                sum_utility = sum_utility + worker.offline_utility
            # else:
                # print('worker {} conflicted with winners.'.format(worker.id))
        winner_list = winner_list[:-1]
        return winner_list

 # -------------------------------online policies-----------------------------------
    def calK(self, worker_list, budget=B):
        sum_cost = 0
        sum_utility = 0
        temp_worker_list = copy.deepcopy(worker_list)
        temp_worker_list = sorted(temp_worker_list, key=lambda x: x.cost, reverse=True)
        winner_list = []
        while (budget > sum_cost):
            worker = temp_worker_list.pop(0)
            if self.ifconflict_2(worker=worker, worker_list=winner_list) == 0:
                winner_list.append(worker)
                sum_cost = sum_cost + worker.cost
                sum_utility = sum_utility + worker.offline_utility
            # else:
                # print('worker {} conflicted with winners.'.format(worker.id))
        winner_list = winner_list[:-1]
        K=len(winner_list)
        return K

    def calCriteria_OACP(self,worker, K, t, worker_list):
        average=np.mean(worker.reward_list)
        c_min = np.max([i.cost for i in worker_list])
        e=math.sqrt((B/c_min+1)*math.log(t, math.e)/len(worker.reward_list))
        return average+e

    def online_policy_OACP(self, worker_list, budget=B, Time=T):
        temp_worker_list=copy.deepcopy(worker_list)
        # K=self.calK(worker_list=temp_worker_list,budget=budget)
        # print('K:{}'.format(K))
        utility_dict={}
        t=1
        while(t < Time):
            if (t%10==0):
                print('----OACP, t:',t)
            # print('t: {}'.format(t))
            for i in temp_worker_list:
                i.offline_utility=self.calCriteria_OACP(worker=i,K=0,t=t,worker_list=temp_worker_list)
            temp_winner_list=self.off_policy_CAP(worker_list=temp_worker_list, budget=budget)
            for winner in temp_winner_list:
                reward=winner.online_utility_list[t]
                while(reward<0):
                    reward = np.random.normal(loc=winner.mean, scale=variance, size=1)[0]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            t=t+1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def online_policy_Optimal(self, worker_list, budget=B, Time=T):
        temp_worker_list = copy.deepcopy(worker_list)
        utility_dict={}
        t = 1
        while (t <= Time):
            # print('t: {}'.format(t))
            temp_winner_list = self.off_policy_TRAC(worker_list=temp_worker_list, budget=budget)
            for winner in temp_winner_list:
                reward = winner.online_utility_list[t]
                while (reward < 0):
                    reward = np.random.normal(loc=winner.mean, scale=winner.variance, size=1)[0]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            t = t + 1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def calCriteria_UCB_MB(self,worker, K, t, worker_list):
        average=np.mean(worker.reward_list)
        c_min=np.min([i.cost for i in worker_list])
        a=math.sqrt((n+1)*math.log(t,math.e)/len(worker.reward_list))
        b=(1+1/c_min)/(c_min-a)
        e=a*b*worker.cost
        return average+e

    def online_policy_UCB_MB(self, worker_list, budget=B, Time=T):
        temp_worker_list=copy.deepcopy(worker_list)
        # K=self.calK(worker_list=temp_worker_list,budget=budget)
        K = budget / np.mean([i.cost for i in temp_worker_list])
        # print('K:{}'.format(K))
        utility_dict={}
        t=1
        while(t <= Time):
            # print('t: {}'.format(t))
            for i in temp_worker_list:
                i.offline_utility=self.calCriteria_UCB_MB(worker=i,K=0,t=t,worker_list=temp_worker_list)
            temp_winner_list=self.off_policy_TRAC_2(worker_list=temp_worker_list, budget=budget)
            for winner in temp_winner_list:
                reward=winner.online_utility_list[t]
                while(reward<0):
                    reward = np.random.normal(loc=winner.mean, scale=winner.variance, size=1)[0]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            # utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list[:-1]])
            t=t+1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def calCriteria_AUCB(self,worker, K, t):
        average=np.mean(worker.reward_list)
        a=math.sqrt((n+1)*math.log(t,math.e)/len(worker.reward_list))
        e=a/worker.cost
        # e=a/worker.cost
        return average + e

    def online_policy_AUCB(self, worker_list, budget=B, Time=T):
        temp_worker_list=copy.deepcopy(worker_list)
        # K=self.calK(worker_list=temp_worker_list,budget=budget)
        K=budget/np.mean([i.cost for i in temp_worker_list])
        # print('K:{}'.format(K))
        utility_dict={}
        t=1
        while(t <= Time):
            # print('t: {}'.format(t))
            for i in temp_worker_list:
                i.offline_utility=self.calCriteria_AUCB(worker=i,K=0,t=t)
            temp_winner_list=self.off_policy_TRAC_2(worker_list=temp_worker_list, budget=budget)
            for winner in temp_winner_list:
                reward=winner.online_utility_list[t]
                while(reward<0):
                    reward = np.random.normal(loc=winner.mean, scale=winner.variance, size=1)[0]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            # utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list[:-1]])
            t=t+1
        sum_utility = sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

    def calCriteria_LLR(self,worker, K, t):
        average=np.mean(worker.reward_list)
        a=math.sqrt((n+1)*math.log(t,math.e)/len(worker.reward_list))
        e=a
        return average + e

    def online_policy_LLR(self, worker_list, budget=B, Time=T):
        temp_worker_list=copy.deepcopy(worker_list)
        # K=self.calK(worker_list=temp_worker_list,budget=budget)
        # print('K:{}'.format(K))
        utility_dict={}
        t=1
        while(t <= Time):
            # print('t: {}'.format(t))
            for i in temp_worker_list:
                i.offline_utility=self.calCriteria_LLR(worker=i,K=0,t=t)
            temp_winner_list=self.off_policy_TRAC(worker_list=temp_worker_list, budget=budget)
            for winner in temp_winner_list:
                reward=winner.online_utility_list[t]
                while(reward<0):
                    reward = np.random.normal(loc=winner.mean, scale=winner.variance, size=1)[0]
                winner.reward_list.append(reward)
            utility_dict[t] = sum([i.reward_list[-1] for i in temp_winner_list])
            t=t+1
        sum_utility=sum([utility_dict[i] for i in utility_dict.keys()])
        return sum_utility, utility_dict

 # ----------------------------evaluatation function-------------------------
#------offline-------
    def run_offline(self, city='Tokyo'):
        dict5 = {}
        for i in range(self.count):
            # print('count:', i)
            worker_list, task_list = self.loadWorkers(city=city)
            u1 = self.off_policy_CAP(worker_list=worker_list, budget=self.B)
            u2 = self.off_policy_random(worker_list=worker_list, budget=self.B)
            u3 = self.off_policy_TRAC(worker_list=worker_list, budget=self.B)
            u4 = self.off_policy_greedy(worker_list=worker_list, budget=self.B)
            u5 = self.off_policy_TRAC_2(worker_list=worker_list, budget=self.B)
            dict5[i] = [sum([i.offline_utility for i in u1]),sum([i.offline_utility for i in u2]),
                        sum([i.offline_utility for i in u3]), sum([i.offline_utility for i in u4]),
                        sum([i.offline_utility for i in u5])]
        u1 = np.mean([list(dict5[i])[0] for i in dict5.keys()])
        u2 = np.mean([list(dict5[i])[1] for i in dict5.keys()])
        u3 = np.mean([list(dict5[i])[2] for i in dict5.keys()])
        u4 = np.mean([list(dict5[i])[3] for i in dict5.keys()])
        u5 = np.mean([list(dict5[i])[4] for i in dict5.keys()])
        return [u1, u2, u3, u4, u5]

    def run_offline_n(self, city='Tokyo'):
        path_utility = result_address+'\offline_n_'+city+'.txt'
        self.reset()
        print('B:{} n:{} m:{}'.format(self.B, self.n, self.m))
        str='%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('n', 'CAP', 'random', 'TRAC', 'greedy','TRAC2')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        n_list = [i for i in range(60,121,10)]
        for x in n_list:
            self.n=x
            utility_list = self.run_offline(city=city)
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.n, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                self.n, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4]))

    def run_offline_B(self, city='Tokyo'):
        path_utility = result_address+'\offline_B_'+city+'.txt'
        self.reset()
        print('B:{} n:{} m:{}'.format(self.B, self.n, self.m))
        str='%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('B', 'CAP', 'random', 'TRAC', 'greedy','TRAC2')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        B_list = [i for i in range(100,701,100)]
        for x in B_list:
            self.B=x
            utility_list = self.run_offline(city=city)
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.B, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                self.B, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4]))

    def run_offline_time(self, city='Tokyo'):
        path_utility = result_address+'\offline_time_'+city+'.txt'
        self.reset()
        print('B:{} n:{} m:{}'.format(self.B, self.n, self.m))
        str='%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('B', 'n', 'time_cap', 'time_opt', 'utility_cap','utility_opt','ratio')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        B_list = [i for i in range(50,151,50)]
        self.n=50
        for x in B_list:
            self.B=x
            worker_list, task_list = self.loadWorkers(city=city)
            t1=time.time()
            u1 = self.off_policy_CAP(worker_list=worker_list, budget=self.B)
            t2 = time.time()
            list2,u2=self.off_policy_Optimal(worker_list=worker_list, budget=self.B)
            t3 = time.time()
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.B, self.n, t2-t1, t3-t2, u1, u2, u1/u2))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                self.B, self.n, t2-t1, t3-t2, u1, u2,u1/u2))


    def run_offline_bundle(self, city='Tokyo'):
        path_utility = result_address+'\offline_bundle_'+city+'.txt'
        self.reset()
        print('B:{} n:{} m:{}'.format(self.B, self.n, self.m))
        str='%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('bundle', 'CAP', 'random', 'TRAC', 'greedy','TRAC2')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        bundle_list = [i for i in range(1,6)]
        for x in bundle_list:
            self.size=x
            utility_list = self.run_offline(city=city)
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.size, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                self.size, utility_list[0], utility_list[1], utility_list[2], utility_list[3], utility_list[4]))

    def run_offline_payment(self, city='Tokyo'):
        path_utility = result_address+'\offline_payment_'+city+'.txt'
        self.reset()
        print('B:{} n:{} m:{}'.format(self.B, self.n, self.m))
        str='%-18s%-18s%-18s%-18s%-18s\n' % ('B', 'n', 'm', 'CPP','Other')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        B_list = [200, 400]
        n_list = [100, 120]
        for x in B_list:
            self.B=x
            for y in n_list:
                self.n=y
                for z in range(30):
                    worker_list, task_list = self.loadWorkers(city=city)
                    winner_list = self.off_policy_CPP(worker_list=worker_list, budget=self.B)
                    for worker in winner_list:
                        with open(path_utility, 'a') as f:
                            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                                self.B, self.n, self.m, worker.payment - worker.cost,
                                worker.other_payment - worker.cost))
                with open(path_utility, 'a') as f:
                    f.write('\n')
            with open(path_utility, 'a') as f:
                f.write('\n')

# ------online-------
    def run_online(self, city='Tokyo'):
        dict_u = {}
        dict_d={}
        list_d=[]
        for i in range(count):
            # print('count:',i)
            worker_list, task_list = self.loadWorkers(city=city)
            u1, d1 = self.online_policy_OACP(worker_list=worker_list, budget=self.B, Time=self.T)
            u2, d2 = self.online_policy_Optimal(worker_list=worker_list, budget=self.B, Time=self.T)
            u3, d3 = self.online_policy_UCB_MB(worker_list=worker_list, budget=self.B, Time=self.T)
            u4, d4 = self.online_policy_AUCB(worker_list=worker_list, budget=self.B, Time=self.T)
            u5, d5 = self.online_policy_LLR(worker_list=worker_list, budget=self.B, Time=self.T)
            dict_u[i] = [u1, u2, u3, u4, u5]
            dict_d[i] = [d1, d2, d3, d4, d5]
            # print('0--------------',d1==d2)
        for j in range(5):
            dic_x={}
            for t in range(1, self.T):
                dic_x[t] = np.mean([dict_d[i][j][t] for i in dict_d.keys()])
            list_d.append(dic_x)
        # print('0.1--------------',list_d[0]==list_d[1])
        u1 = np.mean([dict_u[i][0] for i in dict_u.keys()])
        u2 = np.mean([dict_u[i][1] for i in dict_u.keys()])
        u3 = np.mean([dict_u[i][2] for i in dict_u.keys()])
        u4 = np.mean([dict_u[i][3] for i in dict_u.keys()])
        u5 = np.mean([dict_u[i][4] for i in dict_u.keys()])
        return [u1,u2,u3,u4,u5], list_d

    def calDict(self, dic, t):
        return sum([dic[i] for i in range(1,t+1)])

    def run_online_B(self, city='Tokyo'):
        self.reset()
        path_utility = result_address+'\online_B_utility_'+city+'.txt'
        path_regret = result_address + '\online_B_regret_'+city+'.txt'
        str='%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('T', 'OCAP',
                                                  'Optimal', 'UCB_MB', 'AUCB','LLR')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)
        B_list = [i for i in range(100, 701, 100)]
        for x in B_list:
            self.B = x
            utility_list, dict_list = self.run_online(city=city)
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                self.B, utility_list[0], utility_list[1], utility_list[2],
                utility_list[3], utility_list[4]))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' %  (
                self.B, utility_list[0], utility_list[1], utility_list[2],
                utility_list[3], utility_list[4]))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' %  (
                self.B, utility_list[1]-utility_list[0], 0, utility_list[1]-utility_list[2],
                utility_list[1]-utility_list[3], utility_list[1]-utility_list[4]))



def run_online_T(self, city='Tokyo'):
        self.reset()
        path_utility = result_address+'\online_T_utility_'+city+'.txt'
        path_regret = result_address + '\online_T_regret_'+city+'.txt'
        str='%-18s%-18s%-18s%-18s%-18s%-18s\n' % ('T', 'OCAP',
                                                  'Optimal', 'UCB_MB', 'AUCB','LLR')
        print(str)
        with open(path_utility, 'a') as f:
            f.write(str)
        with open(path_regret, 'a') as f:
            f.write(str)
        utility_list, dict_list = self.run_online(city=city)
        for t in range(5,self.T,5):
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                t, self.calDict(dict_list[0],t), self.calDict(dict_list[1],t), self.calDict(dict_list[2],t),
                self.calDict(dict_list[3],t), self.calDict(dict_list[4],t)))
            with open(path_utility, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                t, self.calDict(dict_list[0],t), self.calDict(dict_list[1],t), self.calDict(dict_list[2],t),
                self.calDict(dict_list[3],t), self.calDict(dict_list[4],t)))
            with open(path_regret, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                t,self.calDict(dict_list[1],t)-self.calDict(dict_list[0],t), 0, self.calDict(dict_list[1],t)-self.calDict(dict_list[2],t),
                self.calDict(dict_list[1],t)-self.calDict(dict_list[3],t), self.calDict(dict_list[1],t)-self.calDict(dict_list[4],t)))


if __name__ == '__main__':
    p=Policy()
    # d=p.calBudget()
    # d=p.caln()
    # p.run_offline_B(city='Tokyo')
    # p.run_offline_n(city='Tokyo')
    # p.run_offline_B(city='New York')
    # p.run_offline_n(city='New York')
    # p.run_online_T(city='Tokyo')
    # p.run_online_T(city='New York')
    # p.run_offline_payment(city='Tokyo')
    # p.run_offline_payment(city='New York')
    # p.run_offline_bundle(city='Tokyo')
    # p.run_offline_bundle(city='New York')
    # p.run_offline_time(city='Tokyo')
    # p.run_online_B(city='Tokyo')
    # p.run_online_B(city='New York')








