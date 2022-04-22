# imports
import torch
from torch.autograd import Variable
from torch import nn
from torch.cuda import random
from torch.nn import Parameter

import numpy as np
from numpy.linalg import norm

import scipy.io as sio

import sklearn.metrics as metrics

import pickle

import random

import sys

import matplotlib.pyplot as plt

import math

import time


usecuda = True #start by setting this to True
usecuda = usecuda and torch.cuda.is_available() #will become false if your gpu is not there or not available

dtype = torch.DoubleTensor

if usecuda: #if the the drivers needed for a GPU are available (from checking above)
    dtype = torch.cuda.FloatTensor #just a different data type to help in the optimization

np.seterr(divide='ignore', invalid='ignore') #to not give warnings/errors when dividing by 0 or NaN



class RobotTrustModel(torch.nn.Module):

    def __init__(self):
        super(RobotTrustModel, self).__init__()

        #beta is not relevant for artificial trust model (see BTM paper for natural trust)
        self.pre_beta_1 = Parameter(dtype(4.0 * np.ones(1)), requires_grad=True)
        self.pre_beta_2 = Parameter(dtype(4.0 * np.ones(1)), requires_grad=True)

        #instead of using lower bounds of 0 and upper bounds of 1, the code is more
        #stable when using the range [-10,10] and then converting back later to [0,1].
        #requires_grad=True means it is part of the gradient computation (i.e., the value should be updated when being optimized)
        self.pre_l_1 = Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True)
        self.pre_u_1 = Parameter(dtype( 10.0 * np.ones(1)), requires_grad=True)
        self.pre_l_2 = Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True)
        self.pre_u_2 = Parameter(dtype( 10.0 * np.ones(1)), requires_grad=True)



    def forward(self, bin_centers, obs_probs_idxs):
        #bin_centers and obs_probs_idxs are passed in
        
        n_diffs = obs_probs_idxs.shape[0] #the number of [row,col] indexes (max is nbins x nbins (625))
        trust = torch.zeros(n_diffs) #create a 1xn_diffs array of 0s

        if(self.pre_l_1 > self.pre_u_1): #if the lower bound is greater than the upper bound
            buf = self.pre_l_1 #switch the l_1 and u_1 values
            self.pre_l_1 = self.pre_u_1
            self.pre_u_1 = buf

        if(self.pre_l_2 > self.pre_u_2): #if the lower bound is greater than the upper bound
            buf = self.pre_l_2 #switch the l_2 and u_2 values
            self.pre_l_2 = self.pre_u_2
            self.pre_u_2 = buf

        l_1 = self.sigm(self.pre_l_1) #convert to [0,1] range
        u_1 = self.sigm(self.pre_u_1)
        beta_1 = self.pre_beta_1 * self.pre_beta_1 #want beta to be positive to compute trust using the artificial trust model

        l_2 = self.sigm(self.pre_l_2)
        u_2 = self.sigm(self.pre_u_2)
        beta_2 = self.pre_beta_2 * self.pre_beta_2


        for i in range(n_diffs): #loop over the number of the [row,col] indexes (max is nbins x nbins (625))
            bin_center_idx_1 = obs_probs_idxs[i, 0] #get the ith cell row
            bin_center_idx_2 = obs_probs_idxs[i, 1] #get the ith cell col
            trust[i] = self.compute_trust(l_1, u_1, beta_1, bin_centers[bin_center_idx_1]) * self.compute_trust(l_2, u_2, beta_2, bin_centers[bin_center_idx_2])
            #computing the trust estimate for each cell based on the current lower and upper bounds (basically the 3d trust plot)

        if usecuda:
            trust = trust.cuda()
        
        return trust

    def compute_trust(self, l, u, b, p):
        #passing in lower bound capability belief, upper bound capability belief, beta, task requirement lambdabar

        if b < -50: #this is for natural trust. This never happens for the artificial trust model.
            trust = 1.0 - 1.0 / (b * (u - l)) * torch.log( (1.0 + torch.exp(b * (p - l))) / (1.0 + torch.exp(b * (p - u))) )
        
        else: #as long as you pass in a positive beta, we will be calculating artificial trust which doesnt depend on beta
            if p <= l: #if lambdabar is less than the lower bound capability belief
                trust = torch.tensor([1.0]) #assign a trust of 1
            elif p > u: #if lambdabar is greater than the upper bound capability belief
                trust = torch.tensor([0.0]) #assign a trust of 0
            else:
                trust = (u - p) / (u - l + 0.0001) #assign trust as a constant slope between u and l. 0.0001 is to not divide by 0.

        if usecuda:
            trust = trust.cuda()
        
        return trust #returns the trust in human agent given lower bound l, upper bound u, beta term b, and task requirement lambdabar p 

    def sigm(self, x): #sigmoid function to convert [-10,10] (really [-inf,inf]) to [0,1]
        return 1 / (1 + torch.exp(-x))

    def sigmoid(self, lambdabar, agent_c):
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
    
        model = RobotTrustModel() #create a RobotTrustModel object

        if usecuda:
            model.cuda()


        num_tasks = 500 #Max = 500 based on .mat file. Change to the number of tasks you want to allocate.

        total_num_tasks = [[num_tasks]]
        p = [] #2 x num_tasks array to hold all task requirements
        p1 = [] #1 x num_tasks array to hold lambdabar1 value (first row of p)
        p2 = [] #1 x num_tasks array to hold lambdabar2 value (second row of p)
        perfs = [] #1 x num_tasks array to hold team successes or failures 

        human_p1 = [] #array to hold the lambdabar1 requirements for tasks assigned to the human
        human_p2 = [] #array to hold the lambdabar2 requirements for tasks assigned to the human
        robot_p1 = [] #array to hold the lambdabar1 requirements for tasks assigned to the robot
        robot_p2 = [] #array to hold the lambdabar2 requirements for tasks assigned to the robot
        tie_p1 = [] #array to hold the lambdabar1 requirements for tasks that are a tie
        tie_p2 = [] #array to hold the lambdabar2 requirements for tasks that are a tie
        assigned = -1 #to indicate which agent the task is assigned to (-1 means no one yet, 0 means robot, 1 means human)
        human_trust = [] #array to hold trust in the human for every task
        robot_trust = [] #array to hold trust in the robot for every task
        human_expected = [] #array to hold expected human reward for every task
        robot_expected = [] #array to hold expected robot reward for every task
        human_perfs = [] #array to hold human successes or failures for tasks assigned to the human
        human_successes = [[],[]] #array to hold the task requirements for tasks the human successed on
        human_failures = [[],[]] #array to hold the task requirements for tasks the human failed on
        robot_perfs = [] #array to hold robot successes or failures for tasks assigned to the robot
        robot_successes = [[],[]] #array to hold the task requirements for tasks the robot successed on
        robot_failures = [[],[]] #array to hold the task requirements for tasks the robot failed on
        human_num_tasks = 0 #the number of tasks assigned to the human
        robot_num_tasks = 0 #the number of tasks assigned to the robot
        tie_num_tasks = 0 #the number of tasks that were initially a tie

       
        total_reward = 0 #the total reward for this task
        max_total_reward = 0 #the max total reward if every task was a success

        #fabricated capability values for the human
        human_l1 = 0.54
        human_u1 = 0.56
        human_l2 = 0.74
        human_u2 = 0.76    
        human_beta1 = 1000 #remember beta does not matter as long as it is positive
        human_beta2 = human_beta1

        #fabricated capability values for the robot
        robot_l1 = 0.69
        robot_u1 = 0.71
        robot_l2 = 0.39
        robot_u2 = 0.41
        robot_beta1 = model.pre_beta_1 * model.pre_beta_1 
        robot_beta2 = model.pre_beta_2 * model.pre_beta_2 

        
        human_c1 = (human_l1 + human_u1)/2.0 #actual capabilities
        human_c2 = (human_l2 + human_u2)/2.0
        robot_c1 = (robot_l1 + robot_u1)/2.0
        robot_c2 = (robot_l2 + robot_u2)/2.0

        load_file_name = "./results/tasks/TA_normaldist_tasks_" + str(iter) + ".mat"
        fixed_tasks_mat = sio.loadmat(load_file_name) #2x500 tasks
        p_from_mat = fixed_tasks_mat["p"] #an array of 2x500 tasks

        #take the tasks for the number of tasks you want
        p = p_from_mat[0:2,0:num_tasks] #take rows 0 to 1 (not including 2) and num_tasks cols from 0 to num_tasks-1 


        #ATTA when the capabilities belief has already converged
        print("Running ATTA Iter " + str(iter))
        for i in range(num_tasks): #iterates from 0 to num_tasks-1 for a total of num_tasks times


            #compute trust in each agent now based on current (converged) belief in lower and upper bounds
            humantrust_i = model.compute_trust(human_l1, human_u1, human_beta1, p[0][i]) * model.compute_trust(human_l2, human_u2, human_beta2, p[1][i])
            robottrust_i = model.compute_trust(robot_l1, robot_u1, robot_beta1, p[0][i]) * model.compute_trust(robot_l2, robot_u2, robot_beta2, p[1][i])
            
            human_trust = np.append(human_trust, humantrust_i.item())
            robot_trust = np.append(robot_trust, robottrust_i.item())

            if usecuda:
                humantrust_i = humantrust_i.cuda()
                robottrust_i = robottrust_i.cuda()
                

            #compute the rewards and costs now
            reward = (p[0][i] + p[1][i])/(2.0) #use this form if the reward is independent of the agent

            humanCost = (p[0][i] + p[1][i])/(3.0)
            robotCost = (p[0][i] + p[1][i])/(8.0)

            Ehuman = humantrust_i.item()*reward - humanCost
            Erobot = robottrust_i.item()*reward - robotCost

            human_expected = np.append(human_expected, Ehuman)
            robot_expected = np.append(robot_expected, Erobot)


            #assign the task now
            assigned = -1 #to indicate the task is not assigned yet
            Ediff = abs(Ehuman - Erobot)
            alpha_tolerance = 0.0 #to help balance the distribution of tasks assigned to each agent
            if Ediff <= alpha_tolerance:
                tie_p1 = np.append(tie_p1, p[0][i])
                tie_p2 = np.append(tie_p2, p[1][i])
                tie_num_tasks = tie_num_tasks + 1 #the number of tasks that were initially tied
                print("not assigned yet: tie")

                if human_num_tasks <= robot_num_tasks:
                    human_p1 = np.append(human_p1, p[0][i]) #append the task's lambdabar1 requirement
                    human_p2 = np.append(human_p2, p[1][i]) #append the task's lambdabar2 requirement
                    assigned = 1
                    human_num_tasks = human_num_tasks + 1
                    print("within tolerance assigned to human")
                else:
                    robot_p1 = np.append(robot_p1, p[0][i]) 
                    robot_p2 = np.append(robot_p2, p[1][i])
                    assigned = 0
                    robot_num_tasks = robot_num_tasks + 1
                    print("within tolerance assigned to robot")
            else:
                if (Ehuman > Erobot):
                    human_p1 = np.append(human_p1, p[0][i]) #append the task's lambda1 requirement
                    human_p2 = np.append(human_p2, p[1][i]) #append the task's lambda2 requirement
                    assigned = 1 #to indicate the task is assigned to the human
                    human_num_tasks = human_num_tasks + 1
                    print("assigned to human")
                    
                elif (Ehuman < Erobot):
                    robot_p1 = np.append(robot_p1, p[0][i]) 
                    robot_p2 = np.append(robot_p2, p[1][i])
                    assigned = 0 #to indicate the task is assigned to the robot
                    robot_num_tasks = robot_num_tasks + 1
                    print("assigned to robot")
            
                #the case of Ehuman == Erobot is captured in Ediff <= alpha_tolerance above


            #observe the task outcome now
            col_i = np.vstack(([p[0][i]],[p[1][i]])) #col for hstack
            tester = random.random()
            perf_i = 0 #failure by default on the ith task

            #compute the true trust probability for each agent
            human_outcome_prob = model.sigmoid(p[0][i], human_c1) * model.sigmoid(p[1][i], human_c2)
            robot_outcome_prob = model.sigmoid(p[0][i], robot_c1) * model.sigmoid(p[1][i], robot_c2)


            if assigned == 1: #if the task was assigned to the human

                if tester <= human_outcome_prob:
                    perf_i = 1 #success on the ith task
                    human_successes = np.hstack((human_successes, col_i))
                    total_reward = total_reward + (reward - humanCost)
                else:
                    human_failures = np.hstack((human_failures, col_i))
                    total_reward = total_reward - humanCost
                human_perfs = np.append(human_perfs, perf_i)
                max_total_reward = max_total_reward + (reward - humanCost)
            
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
            
            else: #if assigned == -1 and the task was not assigned to anyone
                raise ValueError("Error: Task not assigned")

            perfs = np.append(perfs, perf_i) #append the performance to the perfs array

        res_dict = {"human_p1": human_p1, "human_p2": human_p2, "robot_p1": robot_p1, "robot_p2": robot_p2, "human_perfs": human_perfs, "robot_perfs": robot_perfs, "human_successes": human_successes, "human_failures": human_failures, "robot_successes": robot_successes, "robot_failures": robot_failures, "human_num_tasks": human_num_tasks, "robot_num_tasks": robot_num_tasks, "tie_num_tasks": tie_num_tasks, "total_num_tasks": total_num_tasks[0][0], "human_trust": human_trust, "robot_trust": robot_trust, "human_expected": human_expected, "robot_expected": robot_expected, "p": p, "perfs": perfs, "human_l1": human_l1, "human_u1": human_u1, "human_l2": human_l2, "human_u2": human_u2, "robot_l1": robot_l1, "robot_u1": robot_u1, "robot_l2": robot_l2, "robot_u2": robot_u2, "total_reward": total_reward, "max_total_reward": max_total_reward}
        res_mat_file_name = "results/atta/atta_caseI_eta50_" + str(iter) + ".mat"
        sio.savemat(res_mat_file_name, res_dict) #save to file   


