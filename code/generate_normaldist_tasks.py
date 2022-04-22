# imports
import numpy as np
from numpy.linalg import norm

import scipy.io as sio

import sklearn.metrics as metrics

import matplotlib.pyplot as plt
from scipy import stats


num_iter = 10

num_tasks = 500 #Change to the number of tasks you want to generate in a set.

saving_new_tasks = True


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

mu = 0
sigma1 = ((max(human_c1, robot_c1))/2.0) + 0.05 #some offsets that generated tasks covering a good portion of the capability hypercube
sigma2 = ((max(human_c2, robot_c2))/2.0) + 0.12

class my_distribution1(stats.rv_continuous):
    def _pdf(self, x):
        return (2/(sigma1 * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma1**2) ))

class my_distribution2(stats.rv_continuous):
    def _pdf(self, x):
        return (2/(sigma2 * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma2**2) ))


# distribution1 = my_distribution1(a=0, b=1, name='my_distribution1') #interval is [0,1]
# distribution2 = my_distribution2(a=0, b=1, name='my_distribution2') #interval is [0,1]
distribution1 = my_distribution1(a=0) #interval for task requirement value is [0,inf]
distribution2 = my_distribution2(a=0) #interval for task requirement value is [0,inf]

for iter in range(0, num_iter, 1):
    p = [] #2D array to hold all task requirements
    p1 = []
    p2 = []
    tasks = 0 #the number of tasks we have generated so far

    while tasks < num_tasks:
        p11 = distribution1.rvs() #sample a random variable from the distribution
        p22 = distribution2.rvs()

        if (p11 >= 0.0 and p11 <= 1.0) and (p22 >= 0.0 and p22 <= 1.0):
            p1.append(p11) #the lambdabar1 task requirement is within [0,1]
            p2.append(p22)
            tasks += 1

    p = np.vstack((p1,p2))


    if saving_new_tasks:
        res_dict = {"p": p, "p1": p1, "p2": p2, "num_tasks": num_tasks, "human_l1": human_l1, "human_u1": human_u1, "human_l2": human_l2, "human_u2": human_u2, "robot_l1": robot_l1, "robot_u1": robot_u1, "robot_l2": robot_l2, "robot_u2": robot_u2}
        res_mat_file_name = "results/tasks/TA_normaldist_tasks_" + str(iter) + ".mat"
        sio.savemat(res_mat_file_name, res_dict) #save to file 






