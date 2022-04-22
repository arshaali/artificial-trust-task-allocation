# imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.cuda import random
from torch.nn import Parameter

import numpy as np
from numpy.linalg import norm

import scipy.io as sio

import random

import matplotlib.pyplot as plt

import math


def sigmoid(lambdabar, agent_c):
    #takes in task requirement for one dimension and the agent's actual capability for that dimension
    #calculates true trust to determine the stochastic task outcome

    #if lambdabar == agent_c, the sigmoid output is 0.5
    eta = 1/50.0 #is a good value through testing for good capability updating
    #eta = 1/5.0
    #eta = 1/500.0
    return 1 / (1 + math.exp((lambdabar - agent_c)/eta))



num_iter = 10 #run the strategy 10 times

if __name__ == "__main__":

    
    for iter in range(0, num_iter, 1):

        num_tasks = 500 #Max = 500 based on .mat file. Change to the number of tasks you want to allocate.

        total_num_tasks = [[num_tasks]]

        p = [] #2D array to hold all task requirements
        p1 = []
        p2 = []
        perfs = [] #array to hold team successes or failures

        human_p1 = [] #array to hold the lambdabar1 requirements for tasks assigned to the human
        human_p2 = [] #array to hold the lambdabar2 requirements for tasks assigned to the human
        robot_p1 = [] #array to hold the lambdabar1 requirements for tasks assigned to the robot
        robot_p2 = [] #array to hold the lambdabar2 requirements for tasks assigned to the robot
        assigned = -1 #to indicate which agent the task is assigned to

        human_perfs = [] #array to hold human successes or failures for tasks assigned to the human
        human_successes = [[],[]] #array to hold the tasks the human successed on
        human_failures = [[],[]] #array to hold the tasks the human failed on
        robot_perfs = []
        robot_successes = [[],[]]
        robot_failures = [[],[]] #array to hold the tasks the robot failed on
        human_num_tasks = 0 #the number of tasks assigned to the human
        robot_num_tasks = 0  

        discarded_p1 = [] #tasks that were not allocated
        discarded_p2 = []
        discarded_num_tasks = 0 #the number of discarded tasks
        discarded_num_robotCost = 0
        discarded_num_humanCost = 0 #the number of times the discarded task incurred the humanCost
    
        total_reward = 0
        max_total_reward = 0

        #fabricated capability values for the human
        human_l1 = 0.54
        human_u1 = 0.56
        human_l2 = 0.74
        human_u2 = 0.76    

        #fabricated capability values for the robot
        robot_l1 = 0.69
        robot_u1 = 0.71
        robot_l2 = 0.39
        robot_u2 = 0.41
        
        human_c1 = (human_l1 + human_u1)/2.0
        human_c2 = (human_l2 + human_u2)/2.0
        robot_c1 = (robot_l1 + robot_u1)/2.0
        robot_c2 = (robot_l2 + robot_u2)/2.0

        human_offset1 = -0.1 #how much the lambda1 value is off/inaccurate by from human_c1
        human_offset2 = -0.1 #how much the lambda2 value is off/inaccurate by from human_c2
        #0.0 means following Tsarouchi et al. with accurate human capabilities (case I)
        #changing the offset means following Tsarouchi et al. with inaccurate capabilities (case II)

        load_file_name = "./results/tasks/TA_normaldist_tasks_" + str(iter) + ".mat"
        fixed_tasks_mat = sio.loadmat(load_file_name) #2x500
        p_from_mat = fixed_tasks_mat["p"] #an array of 2x500 tasks

        #take the tasks for the number of tasks you want
        p = p_from_mat[0:2,0:num_tasks] #take rows 0 to 1 (not including 2) and cols 0 to num_tasks-1

        #print("p = ", p)
        #print("p[0][0] = ", p[0][0])
        #print("p[1][0] = ", p[1][0])

        #Tsarouchi et al.
        print("Running Tsarouchi et al. Iter " + str(iter))

        for i in range(num_tasks):
            assigned = -1 #to indicate the task is not assigned yet

            #compute the task reward and agent costs now for calculating the total_reward
            reward = (p[0][i] + p[1][i])/(2.0)

            humanCost = (p[0][i] + p[1][i])/(3.0)
            robotCost = (p[0][i] + p[1][i])/(8.0)

            if (p[0][i] <= (human_c1 + human_offset1)) and (p[1][i] <= (human_c2 + human_offset2)) and (p[1][i] > robot_c2): #only the human is capable of this task
                human_p1 = np.append(human_p1, p[0][i]) #append the task's lambdabar1 requirement
                human_p2 = np.append(human_p2, p[1][i]) #append the task's lambdabar2 requirement
                assigned = 1 #to indicate the task is assigned to the human
                human_num_tasks = human_num_tasks + 1
                print("assigned to human")                

            elif (p[0][i] > (human_c1 + human_offset1)) and (p[0][i] <= robot_c1) and (p[1][i] <= robot_c2): #only the robot is capable of this task
                robot_p1 = np.append(robot_p1, p[0][i]) 
                robot_p2 = np.append(robot_p2, p[1][i])
                assigned = 0 #to indicate the task is assigned to the robot
                robot_num_tasks = robot_num_tasks + 1
                print("assigned to robot")

            elif (p[0][i] > (robot_c1)) or (p[1][i] > (human_c2 + human_offset2)) or ((p[0][i] > (human_c1 + human_offset1)) and (p[1][i] > robot_c2)): #no agent is capable
                discarded_p1 = np.append(discarded_p1, p[0][i]) 
                discarded_p2 = np.append(discarded_p2, p[1][i])
                assigned = -2 #to indicate the task is discarded
                discarded_num_tasks = discarded_num_tasks + 1
                print("discarded")

            else: #both agents can are capable
                #assign to agent with lower cost (proxy for operation time)
                if robotCost < humanCost:
                    robot_p1 = np.append(robot_p1, p[0][i]) 
                    robot_p2 = np.append(robot_p2, p[1][i])
                    assigned = 0 #to indicate the task is assigned to the robot
                    robot_num_tasks = robot_num_tasks + 1
                    print("assigned to robot")
                elif humanCost < robotCost:
                    human_p1 = np.append(human_p1, p[0][i]) #append the task's lambdabar1 requirement
                    human_p2 = np.append(human_p2, p[1][i]) #append the task's lambdabar2 requirement
                    assigned = 1 #to indicate the task is assigned to the human
                    human_num_tasks = human_num_tasks + 1
                    print("assigned to human")
                elif humanCost == robotCost:
                    #costs are the same so randomly assign to one agent as implied in the paper
                    assigned = random.randint(0, 1)
                    if assigned == 0: #to indicate the task is assigned to the robot
                        robot_p1 = np.append(robot_p1, p[0][i]) 
                        robot_p2 = np.append(robot_p2, p[1][i])
                        robot_num_tasks = robot_num_tasks + 1
                        print("assigned to robot")
                    elif assigned == 1: #to indicate the task is assigned to the human
                        human_p1 = np.append(human_p1, p[0][i]) #append the task's lambdabar1 requirement
                        human_p2 = np.append(human_p2, p[1][i]) #append the task's lambdabar2 requirement
                        human_num_tasks = human_num_tasks + 1
                        print("assigned to human")



            #observe the task outcome now
            col_i = np.vstack(([p[0][i]],[p[1][i]])) #col for hstack
            tester = random.random()
            perf_i = 0 #failure by default on the ith task

            #compute the true trust probability for each agent
            human_outcome_prob = sigmoid(p[0][i], human_c1) * sigmoid(p[1][i], human_c2)
            robot_outcome_prob = sigmoid(p[0][i], robot_c1) * sigmoid(p[1][i], robot_c2)



            if assigned == 1: #if the task was assigned to the human

                if tester <= human_outcome_prob:
                    perf_i = 1 #success on the ith task
                    human_successes = np.hstack((human_successes, col_i))
                    total_reward = total_reward + (reward - humanCost)
                else:
                    human_failures = np.hstack((human_failures, col_i))
                    total_reward = total_reward - humanCost
                human_perfs = np.append(human_perfs, perf_i)
                max_total_reward = max_total_reward + (reward - humanCost) #keep track of the max reward we could get if every task outcome was a success
            
            elif assigned == 0:  #if the task was assigned to the robot
                
                if tester <= robot_outcome_prob:
                    perf_i = 1
                    robot_successes = np.hstack((robot_successes, col_i))
                    total_reward = total_reward + (reward - robotCost)
                else:
                    robot_failures = np.hstack((robot_failures, col_i))
                    total_reward = total_reward - robotCost
                robot_perfs = np.append(robot_perfs, perf_i)
                max_total_reward = max_total_reward + (reward - robotCost)

            elif assigned == -2: #if the task was discarded
                
                tester2 = random.randint(0, 1) #to randomly choose either to incur the humanCost or the robotCost for failure
                if tester2 == 0:
                    total_reward = total_reward - robotCost
                    max_total_reward = max_total_reward + (reward - robotCost)
                    robot_num_tasks = robot_num_tasks + 1
                    discarded_num_robotCost = discarded_num_robotCost + 1
                elif tester2 == 1:
                    total_reward = total_reward - humanCost
                    max_total_reward = max_total_reward + (reward - humanCost)
                    human_num_tasks = human_num_tasks + 1
                    discarded_num_humanCost = discarded_num_humanCost + 1
                    
            
            else: #if assigned == -1 and the task was neither assigned nor discarded
                raise ValueError("Error: Task not assigned")

            perfs = np.append(perfs, perf_i) #append the performance to the perfs array

        res_dict = {"human_p1": human_p1, "human_p2": human_p2, "robot_p1": robot_p1, "robot_p2": robot_p2, "discarded_p1": discarded_p1, "discarded_p2": discarded_p2, "human_perfs": human_perfs, "robot_perfs": robot_perfs, "human_successes": human_successes, "human_failures": human_failures, "robot_successes": robot_successes, "robot_failures": robot_failures, "human_num_tasks": human_num_tasks, "robot_num_tasks": robot_num_tasks, "discarded_num_tasks": discarded_num_tasks, "total_num_tasks": total_num_tasks[0][0], "p": p, "perfs": perfs, "human_l1": human_l1, "human_u1": human_u1, "human_l2": human_l2, "human_u2": human_u2, "robot_l1": robot_l1, "robot_u1": robot_u1, "robot_l2": robot_l2, "robot_u2": robot_u2, "total_reward": total_reward, "max_total_reward": max_total_reward, "discarded_num_humanCost": discarded_num_humanCost, "discarded_num_robotCost": discarded_num_robotCost}
        res_mat_file_name = "results/tsarouchi/tsarouchi_caseIIneg10_eta50_" + str(iter) + ".mat"
        sio.savemat(res_mat_file_name, res_dict) #save to file 
