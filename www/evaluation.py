from www.Player import Player
import numpy as np
import pandas as pd
from www.conf import *
import math
import matplotlib.pyplot as plt
import seaborn as sns
# from pure_ldp.frequency_oracles.local_hashing import LHClient, LHServer
from matplotlib.pyplot import MultipleLocator

'''return rewards and regrets of a series of approaches'''
def run(K=Yahoo_K,N=Yahoo_N,B=Yahoo_B,pb1=Yahoo_pb1,pb2=Yahoo_pb2,count=Yahoo_count):
    player=Player()
    player.B=B
    player.N = N
    player.K = K
    player.pb1 = pb1
    player.pb2 = pb2
    player.count=count
    l1=[]
    l2=[]
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    l7 = []
    l8 = []
    l9 = []
    for i in range(player.count):
        player.create_arm_list()
        r1, c1, t1, dic1 = player.policy_bpp()
        r2, c2, t2, dic2 = player.policy_optimal()
        r3, c3, t3, dic3 = player.policy_aucb()
        r4, c4, t4, dic4 = player.policy_ucb_mb()
        r5, c5, t5, dic5 = player.policy_no_privacy()
        r6, c6, t6, dic6 = player.policy_exploitation()
        r7, c7, t7, dic7 = player.policy_exploration()
        r8, c8, t8, dic8 = player.policy_no_privacy_aucb()
        r9, c9, t9, dic9 = player.policy_CUCB_DP()
        l1.append(r1)
        l2.append(r2)
        l3.append(r3)
        l4.append(r4)
        l5.append(r5)
        l6.append(r6)
        l7.append(r7)
        l8.append(r8)
        l9.append(r9)
    list_t = [t1, t2, t3, t4, t5, t6, t7, t8, t9]
    rewards_list=[np.mean(l1), np.mean(l2), np.mean(l3), np.mean(l4), np.mean(l5), np.mean(l6), np.mean(l7), np.mean(l8), np.mean(l9)]
    regrets_list=[np.mean(l2)-i for i in rewards_list]
    rewards_error=[np.std(l1), np.std(l2), np.std(l3), np.std(l4), np.std(l5), np.std(l6), np.std(l7), np.std(l8), np.std(l9)]
    regrets_error=[np.std([i-j for i,j in zip(l2,l1)]),np.std([i-j for i,j in zip(l2,l2)]),np.std([i-j for i,j in zip(l2,l3)]),np.std([i-j for i,j in zip(l2,l4)]),
                   np.std([i-j for i,j in zip(l2,l5)]),np.std([i-j for i,j in zip(l2,l6)]),np.std([i-j for i,j in zip(l2,l7)]),np.std([i-j for i,j in zip(l2,l8)]),np.std([i-j for i,j in zip(l2,l9)])]
    return rewards_list, regrets_list, rewards_error, regrets_error, list_t

def run_continue(K=Yahoo_K,N=Yahoo_N,B=Yahoo_B,pb1=Yahoo_pb1,pb2=Yahoo_pb2,count=Yahoo_count):
    player=Player()
    player.B=B
    player.N = N
    player.K = K
    player.pb1 = pb1
    player.pb2 = pb2
    player.count=count
    l1=[]
    l2=[]
    l3 = []
    l4 = []
    l5 = []
    l6 = []
    l7 = []
    l8 = []
    l9 = []
    for i in range(player.count):
        player.create_arm_list()
        r1, c1, t1, dic1 = player.policy_bpp()
        r2, c2, t2, dic2 = player.policy_optimal()
        r3, c3, t3, dic3 = player.policy_aucb()
        r4, c4, t4 , dic4= player.policy_ucb_mb()
        r5, c5, t5 , dic5= player.policy_no_privacy()
        r6, c6, t6 , dic6= player.policy_exploitation()
        r7, c7, t7 , dic7= player.policy_exploration()
        r8, c8, t8 , dic8= player.policy_no_privacy_aucb()
        r9, c9, t9, dic9 = player.policy_CUCB_DP()
        l1.append(dic1)
        l2.append(dic2)
        l3.append(dic3)
        l4.append(dic4)
        l5.append(dic5)
        l6.append(dic6)
        l7.append(dic7)
        l8.append(dic8)
        l9.append(dic9)
    list_t = [t1, t2, t3, t4, t5, t6, t7, t8, t9]
    return l1,l2,l3,l4,l5,l6,l7,l8,l9, count, list_t

def run_only_bpp(K=Yahoo_K,N=Yahoo_N,B=Yahoo_B,pb1=Yahoo_pb1,pb2=Yahoo_pb2,count=Yahoo_count):
    player=Player()
    player.B=B
    player.N = N
    player.K = K
    player.pb1 = pb1
    player.pb2 = pb2
    player.count=count
    l1=[]
    l2=[]
    l3 = []
    for i in range(player.count):
        player.create_arm_list()
        r1, c1, t1, dic1 = player.policy_bpp()
        r2, c2, t2, dic2 = player.policy_bpp_no_private_index()
        r3, c3, t3, dic3 = player.policy_bpp_no_exploration_index()
        l1.append(r1)
        l2.append(r2)
        l3.append(r3)
    rewards_list=[np.mean(l1), np.mean(l2), np.mean(l3)]
    regrets_list=[np.mean(l2)-i for i in rewards_list]
    rewards_error=[np.std(l1), np.std(l2), np.std(l3)]
    regrets_error=[np.std([i-j for i,j in zip(l2,l1)]),np.std([i-j for i,j in zip(l2,l2)]),np.std([i-j for i,j in zip(l2,l3)])]
    return rewards_list, regrets_list, rewards_error,regrets_error

''' return a single regret of bpp'''
def run_single(K=Yahoo_K,N=Yahoo_N,B=Yahoo_B,pb1=Yahoo_pb1,pb2=Yahoo_pb2,count=Yahoo_count):  # return a single regret of bpp
    player=Player()
    player.B=B
    player.N = N
    player.K = K
    player.pb1 = pb1
    player.pb2 = pb2
    player.count=count
    l1=[]
    l2=[]
    for i in range(player.count):
        player.create_arm_list()
        r1, c1, t1, dic1 = player.policy_bpp()
        r2, c2, t1, dic1 = player.policy_optimal()
        l1.append(r1)
        l2.append(r2)
    regret=np.mean(l2)-np.mean(l1)
    return regret




#-----------------specific batch experimental function-----------------
'''K is increased'''
def run_K_rewards():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/K_rewards.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/K_regrets.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'K', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_rewards, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'K', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'K', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    K_list=[1+i for i in range(1,20)]
    '''2,11,12,...,20'''
    for K in K_list:
        rewards_list,regrets_list,rewards_error,regrets_error, list_t=run(K=K)
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
            K, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                K, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                K, regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
            , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]))


'''N is increased'''
def run_N_rewards():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/N_rewards.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/N_regrets.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'N', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_rewards, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'N', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'N', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    N_list=[i for i in range(10,37,1)]
    '''10~36'''
    for N in N_list:
        rewards_list,regrets_list,rewards_error,regrets_error, list_t=run(N=N)
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
            N, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                N, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                N, regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
            , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]))


'''N is increased only with no e-index and p-index'''
def run_N_rewards_only_bpp():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/N_rewards_only_bpp.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/N_regrets_only_bpp.txt'
    print('%-18s%-18s%-18s%-18s' % (
        'N', 'bpp', 'bpp_no_p', 'bpp_no_e'))
    with open(path_rewards, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s\n' % (
        'N', 'bpp', 'bpp_no_p', 'bpp_no_e'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s\n' % (
        'N', 'bpp', 'bpp_no_p', 'bpp_no_e'))
    N_list=[50+i*5 for i in range(21)]
    '''50,55,60,...,150'''
    for N in N_list:
        rewards_list,regrets_list,rewards_error,regrets_error=run_only_bpp(N=N)
        print('%-18.2f%-18.2f%-18.2f%-18.2f' % (
            N, rewards_list[0], rewards_list[1], rewards_list[2]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
            N, rewards_list[0], rewards_list[1], rewards_list[2]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
            N, rewards_list[0], rewards_list[1], rewards_list[2]))


'''B is increased'''
def run_B_rewards():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/budget_rewards.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/budget_regrets.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'Budget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_rewards, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'Budget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'Budget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    B_list=[i*100 for i in range(6,61)]
    '''1000,1500,2000,...,10000'''
    for B in B_list:
        rewards_list,regrets_list,rewards_error,regrets_error, list_t=run(B=B)
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
            B, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f step' % (
            B, list_t[0], list_t[1], list_t[2], list_t[3]
            , list_t[4], list_t[5], list_t[6], list_t[7], list_t[8]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                B, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                B, regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
            , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]))

'''B is increased'''
def run_logB_rewards():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/logB_rewards.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/logB_regrets.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'logBudget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_rewards, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'logBudget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'logBudget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    B_max=9000
    '''600,700,800,...,6000'''
    l1,l2,l3,l4,l5,l6,l7,l8,l9,count,list_t = run_continue(B=B_max)
    if len(l1)!=count:
        print("ERROR: run_countinue, len(l1)!=count")
    for B in range(500,B_max+100,100):
        rewards_list=[]
        temp_list=[l1[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        temp_list_o = [l2[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list_o))
        temp_list = [l3[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        temp_list = [l4[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        temp_list = [l5[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        temp_list = [l6[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        temp_list = [l7[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        temp_list = [l8[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        temp_list = [l9[i][B] for i in range(count)]
        rewards_list.append(np.mean(temp_list))
        regrets_list=[np.mean(temp_list_o)-i for i in rewards_list]
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
            B, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f step' % (
            B, list_t[0], list_t[1], list_t[2], list_t[3]
            , list_t[4], list_t[5], list_t[6], list_t[7], list_t[8]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                B, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
                , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                B, regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
                , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]))


def run_B_rewards_large():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/budget_rewards_large.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/budget_regrets_large.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'Budget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_rewards,  'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'Budget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'Budget', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    B_list=[i*5000 for i in range(1,21)]
    '''1000,1500,2000,...,10000'''
    for B in B_list:
        rewards_list,regrets_list,rewards_error,regrets_error, list_t=run(B=B)
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
            B, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                B, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                B, regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
            , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]))

'''pb_1 is increased'''
def run_pb1_rewards():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/pb1_rewards.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/pb1_regrets.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'pb1', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_rewards, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'pb1', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'pb1', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    pb1_list=[i/10 for i in range(1,21)]
    df=pd.DataFrame(columns={'methods','pb1','reward','reward_error','regret','regret_error'})
    method_list=['bpp', 'optimal', 'aucb', 'ucb_mb', 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP']
    for pb1 in pb1_list:
        rewards_list,regrets_list,rewards_error,regrets_error, list_t=run(pb1=pb1)
        new_df=pd.DataFrame({'methods':method_list,'pb1':[pb1 for i in method_list],
                             'reward':[rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]],
                             'reward_error':[rewards_error[0], rewards_error[1], rewards_error[2], rewards_error[3]
            , rewards_error[4], rewards_error[5], rewards_error[6], rewards_error[7], rewards_error[8]],
                             'regret':[regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
            , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]],
                             'regret_error':[regrets_error[0], regrets_error[1], regrets_error[2], regrets_error[3]
            , regrets_error[4], regrets_error[5], regrets_error[6], regrets_error[7], regrets_error[8]]})
        df=df.append(new_df,ignore_index=True)
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
            pb1, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                pb1, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                pb1, regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
            , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]))
    # df.to_csv('D:/OneDrive/文档/latex/第七篇/实验/pb1_rewards.csv',index=False)

'''pb_2 is increased'''
def run_pb2_rewards():
    path_rewards = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/pb2_rewards.txt'
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/pb2_regrets.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'pb2', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_rewards, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'pb2', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
            'pb2', 'bpp', 'optimal', 'aucb', 'ucb_mb'
        , 'no_privacy_ucb_mb', 'exploitation', 'exploration', 'no_privacy_bpp', 'CUCB_DP'))
    pb2_list=[i/10 for i in range(1,21)]
    for pb2 in pb2_list:
        rewards_list,regrets_list,rewards_error,regrets_error, list_t=run(pb2=pb2)
        print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
            pb2, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_rewards, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                pb2, rewards_list[0], rewards_list[1], rewards_list[2], rewards_list[3]
            , rewards_list[4], rewards_list[5], rewards_list[6], rewards_list[7], rewards_list[8]))
        with open(path_regrets, 'a') as f:
            f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                pb2, regrets_list[0], regrets_list[1], regrets_list[2], regrets_list[3]
            , regrets_list[4], regrets_list[5], regrets_list[6], regrets_list[7], regrets_list[8]))

'''BPP_B_regret_NB'''
def run_NB_regrets():
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/NB_regrets.txt'
    print('%-18s%-18s%-18s' % (
        'budget','N', 'bpp'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s\n' % (
        'budget','N', 'bpp'))
    B_list=[i*1000 for i in range(5,11)]
    B_list =[i for i in range(1000,9000+500,1000)] # [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    N_list = [50 + i * 10 for i in range(6)]
    N_list = [i for i in range(10,37,2)]
    for B in B_list:
        for N in N_list:
            regret = run_single(B=B,N=N)
            print('%-18.2f%-18.2f%-18.2f' % (
                B, N, regret))
            with open(path_regrets, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f\n' % (
                B, N, regret))

'''BPP_B_regret_pb1_pb2'''
def  run_pb1pb2_regrets():
    path_regrets = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/pb1pb2_regrets.txt'
    print('%-18s%-18s%-18s' % (
        'pb1','pb2', 'bpp'))
    with open(path_regrets, 'a') as f:
        f.write('%-18s%-18s%-18s\n' % (
        'pb1','pb2', 'bpp'))
    pb1_list = [i/10 for i in range(1,21)]
    pb2_list = [i/10 for i in range(1,21)]
    for pb1 in pb1_list:
        for pb2 in pb2_list:
            regret = run_single(pb1=pb1, pb2=pb2, B=Yahoo_B,count=Yahoo_count)
            print('%-18.2f%-18.2f%-18.2f' % (
                pb1, pb2, regret))
            with open(path_regrets, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f\n' % (
                pb1, pb2, regret))

def run_payment(K=Yahoo_K,N=Yahoo_N,B=Yahoo_B,pb1=Yahoo_pb1,pb2=Yahoo_pb2,count=Yahoo_count,delta=1):  # return a single regret of bpp
    player=Player()
    player.B=B
    player.N = N
    player.K = K
    player.pb1 = pb1
    path = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/payment.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'delta', 'pb2', 'p1','e1','p2','e2','p1-p2','e*Delta'))
    with open(path, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
        'delta', 'pb2', 'p1','e1','p2','e2','p1-p2','e*Delta'))
    pb2_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    delta_list1=[-2,-1.5,-1,-0.5,0,0.5,1,1,5,2]
    delta_list2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    delta_list = [i for i in delta_list2]
    for delta in delta_list:
        for pb2 in pb2_list:
            player.pb2 = pb2
            player.create_arm_list()
            p1,e1,p2,e2,p_max = player.policy_payment(delta=delta)
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                delta, pb2, p1,e1,p2,e2,p1-p2,p_max*pb2))
            with open(path, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                delta, pb2, p1,e1,p2,e2,p1-p2,p_max*pb2))

def run_payment_loser(K=Yahoo_K,N=Yahoo_N,B=Yahoo_B,pb1=Yahoo_pb1,pb2=Yahoo_pb2,count=Yahoo_count,delta=1):  # return a single regret of bpp
    player=Player()
    player.B=B
    player.N = N
    player.K = K
    player.pb1 = pb1
    path = 'D:/OneDrive/文档/latex/第七篇/实验/Yahoo/payment_loser.txt'
    print('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s' % (
        'delta', 'pb2', 'p1','e1','p2','e2','p1-p2','e*Delta'))
    with open(path, 'a') as f:
        f.write('%-18s%-18s%-18s%-18s%-18s%-18s%-18s%-18s\n' % (
        'delta', 'pb2', 'p1','e1','p2','e2','p1-p2','e*Delta'))
    pb2_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    delta_list1=[-2,-1.5,-1,-0.5,0,0.5,1,1,5,2]
    delta_list2 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    delta_list = [i for i in delta_list2]
    for delta in delta_list:
        for pb2 in pb2_list:
            player.pb2 = pb2
            player.create_arm_list()
            p1,e1,p2,e2,p_max = player.policy_payment(delta=delta)
            print('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f' % (
                delta, pb2, p1,e1,p2,e2,p1-p2,p_max*pb2))
            with open(path, 'a') as f:
                f.write('%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f%-18.2f\n' % (
                delta, pb2, p1,e1,p2,e2,p1-p2,p_max*pb2))



# run_logB_rewards()#new
run_B_rewards()
# run_B_rewards_large()
# run_K_rewards()#new
# run_N_rewards()#new
# run_pb1_rewards()#new
# run_pb2_rewards()#new
# run_NB_regrets()
run_pb1pb2_regrets()
run_payment()
# run_payment_loser()
# run_N_rewards_only_bpp()


