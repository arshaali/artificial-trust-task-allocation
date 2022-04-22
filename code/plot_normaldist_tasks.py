import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#this script is just for plotting the distribution and histogram for tasks that we already have
#this script plots for tasks generated from generate_normaldist_tasks.py

num_tasks = 500


human_l1 = 0.54
human_u1 = 0.56
human_l2 = 0.74
human_u2 = 0.76   
robot_l1 = 0.69
robot_u1 = 0.71
robot_l2 = 0.39
robot_u2 = 0.41


human_c1 = (human_l1 + human_u1)/2.0
human_c2 = (human_l2 + human_u2)/2.0
robot_c1 = (robot_l1 + robot_u1)/2.0
robot_c2 = (robot_l2 + robot_u2)/2.0

mu = 0
sigma1 = ((max(human_c1, robot_c1))/2.0) + 0.05
sigma2 = ((max(human_c2, robot_c2))/2.0) + 0.12


fixed_tasks_mat = sio.loadmat('./results/tasks/TA_normaldist_tasks_3.mat')
p_from_mat = fixed_tasks_mat["p"]
p = p_from_mat[0:2,0:num_tasks]
p1 = p[0,0:num_tasks]
p2 = p[1,0:num_tasks]

count, bins, ignored = plt.hist(p1, 30, density=True)
plt.plot(bins, 2/(sigma1 * np.sqrt(2 * np.pi)) * 
            np.exp( - (bins - mu)**2 / (2 * sigma1**2) ),
        linewidth=2, color='k')
#print(sum(count * bins[1]-bins[0])) #sum( density of each bin * same bin width) should be approx. 1
plt.vlines(x=human_c1, ymin=0, ymax=2.5, colors='b', linestyles='solid', label="Human Capability")
plt.vlines(x=robot_c1, ymin=0, ymax=2.5, colors='r', linestyles='solid', label="Robot Capability")
title = "Histogram and Sampling Distribution for $\lambda_1$"
plt.title(title,fontsize=12)
plt.xlabel("Task $\lambda_1$ Requirement",fontsize=12)
plt.ylabel("Probability Density",fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.legend() #uncomment to see the legend
plt.show()

count, bins, ignored = plt.hist(p2, 30, density=True)
plt.plot(bins, 2/(sigma2 * np.sqrt(2 * np.pi)) *
            np.exp( - (bins - mu)**2 / (2 * sigma2**2) ),
        linewidth=2, color='k')
#print(sum(count * bins[1]-bins[0]))
plt.vlines(x=human_c2, ymin=0, ymax=2.5, colors='b', linestyles='solid', label="Human Capability")
plt.vlines(x=robot_c2, ymin=0, ymax=2.5, colors='r', linestyles='solid', label="Robot Capability")
title = "Histogram and Sampling Distribution for $\lambda_2$"
plt.title(title,fontsize=12)
plt.xlabel("Task $\lambda_2$ Requirement",fontsize=12)
plt.ylabel("Probability Density",fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.legend(loc='best', ncol=2) #uncomment to see the legend
#plt.ylim(-2, 4) #just changed this to get the legend for making the plot with both distributions
plt.show()

#the bar on top of the lambdas was added manually in inkscape
#the plots were combined and the legend added to generate one figure in inkscape
#when the plots are generated, use the save button to save as a .svg or other extension