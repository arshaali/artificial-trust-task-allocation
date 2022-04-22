#imports
import scipy.io as sio
import csv
import pandas as pd 
import numpy as np


num_iter = 10 #number of times the strategies were run

if __name__ == "__main__":
    
    dataprep = []

    for iter in range(0, num_iter, 1):

        load_file_name = "./results/atta/atta_caseII_eta50_" + str(iter) + ".mat"
        output_atta_mat = sio.loadmat(load_file_name)
        
        human_successes_from_atta = output_atta_mat["human_successes"] #tasks assigned to the human that were successes
        human_failures_from_atta = output_atta_mat["human_failures"] #tasks assigned to the human that were failures
        robot_successes_from_atta = output_atta_mat["robot_successes"] #tasks assigned to the robot that were successes
        robot_failures_from_atta = output_atta_mat["robot_failures"] #tasks assigned to the robot that were failures
        total_reward_from_atta = output_atta_mat["total_reward"][0][0]
        max_total_reward_from_atta = output_atta_mat["max_total_reward"][0][0]
        human_num_tasks_from_atta = output_atta_mat["human_num_tasks"][0][0]
        robot_num_tasks_from_atta = output_atta_mat["robot_num_tasks"][0][0]
        total_num_tasks_from_atta = output_atta_mat["total_num_tasks"][0][0]
        l_1 = output_atta_mat["l_1"][0]
        u_1 = output_atta_mat["u_1"][0]
        l_2 = output_atta_mat["l_2"][0]
        u_2 = output_atta_mat["u_2"][0]
        task_number = output_atta_mat["task_number"][0]
        human_l1 = output_atta_mat["human_l1"][0][0]
        human_u1 = output_atta_mat["human_u1"][0][0]
        human_l2 = output_atta_mat["human_l2"][0][0]
        human_u2 = output_atta_mat["human_u2"][0][0]


        l1 = np.zeros(human_num_tasks_from_atta + 1)
        u1 = np.zeros(human_num_tasks_from_atta + 1)
        l2 = np.zeros(human_num_tasks_from_atta + 1)
        u2 = np.zeros(human_num_tasks_from_atta + 1)

        l1[0] = 0 #lower bounds start at 0
        u1[0] = 1 #upper bounds start at 1
        l2[0] = 0
        u2[0] = 1
        j = 1
        for i in range(len(task_number) - 1): #minus 1
            if task_number[i] != task_number[i+1]:

                l1[j] = l_1[i]
                u1[j] = u_1[i]
                l2[j] = l_2[i]
                u2[j] = u_2[i]
                j = j + 1
        
        l1[j] = l_1[-1]
        u1[j] = u_1[-1]
        l2[j] = l_2[-1]
        u2[j] = u_2[-1]

        human_c1 = (human_l1 + human_u1)/2.0
        human_c2 = (human_l2 + human_u2)/2.0


        team_success_rate_atta = 100 * ((human_successes_from_atta.shape[1] + robot_successes_from_atta.shape[1])/(human_successes_from_atta.shape[1] + human_failures_from_atta.shape[1] + robot_successes_from_atta.shape[1] + robot_failures_from_atta.shape[1]))
        human_success_rate_atta = 100 * ((human_successes_from_atta.shape[1])/(human_successes_from_atta.shape[1] + human_failures_from_atta.shape[1]))
        robot_success_rate_atta = 100 * ((robot_successes_from_atta.shape[1])/(robot_successes_from_atta.shape[1] + robot_failures_from_atta.shape[1]))
        team_total_reward_atta = 100 * (total_reward_from_atta/max_total_reward_from_atta)
        perct_human_tasks_atta = 100 * (human_num_tasks_from_atta/total_num_tasks_from_atta)
        perct_robot_tasks_atta = 100 * (robot_num_tasks_from_atta/total_num_tasks_from_atta)

        convergence_lambda1_0 = abs(l1[0]-human_c1) + abs(u1[0]-human_c1)
        convergence_lambda1_5 = abs(l1[5]-human_c1) + abs(u1[5]-human_c1)
        convergence_lambda1_10 = abs(l1[10]-human_c1) + abs(u1[10]-human_c1)
        convergence_lambda1_20 = abs(l1[20]-human_c1) + abs(u1[20]-human_c1)
        convergence_lambda1_40 = abs(l1[40]-human_c1) + abs(u1[40]-human_c1)
        convergence_lambda1_80 = abs(l1[80]-human_c1) + abs(u1[80]-human_c1)
        convergence_lambda1_end = abs(l1[human_num_tasks_from_atta]-human_c1) + abs(u1[human_num_tasks_from_atta]-human_c1)

        convergence_lambda2_0 = abs(l2[0]-human_c2) + abs(u2[0]-human_c2)
        convergence_lambda2_5 = abs(l2[5]-human_c2) + abs(u2[5]-human_c2)
        convergence_lambda2_10 = abs(l2[10]-human_c2) + abs(u2[10]-human_c2)
        convergence_lambda2_20 = abs(l2[20]-human_c2) + abs(u2[20]-human_c2)
        convergence_lambda2_40 = abs(l2[40]-human_c2) + abs(u2[40]-human_c2)
        convergence_lambda2_80 = abs(l2[80]-human_c2) + abs(u2[80]-human_c2)
        convergence_lambda2_end = abs(l2[human_num_tasks_from_atta]-human_c2) + abs(u2[human_num_tasks_from_atta]-human_c2)


        row = [team_success_rate_atta, human_success_rate_atta, robot_success_rate_atta, team_total_reward_atta, perct_human_tasks_atta, perct_robot_tasks_atta, convergence_lambda1_0, convergence_lambda1_5, convergence_lambda1_10, convergence_lambda1_20, convergence_lambda1_40, convergence_lambda1_80, convergence_lambda1_end, convergence_lambda2_0, convergence_lambda2_5, convergence_lambda2_10, convergence_lambda2_20, convergence_lambda2_40, convergence_lambda2_80, convergence_lambda2_end]
        dataprep.append(row)

    df = pd.DataFrame(dataprep, columns=['ATTA Team Perf', 'ATTA Human Perf', 'ATTA Robot Perf', 'ATTA Team Total Reward', 'ATTA Perct Human Tasks', 'ATTA Perct Robot Tasks', 'Convergence lambda1 kH=0', 'Convergence lambda1 kH=5', 'Convergence lambda1 kH=10', 'Convergence lambda1 kH=20', 'Convergence lambda1 kH=40', 'Convergence lambda1 kH=80', 'Convergence lambda1 kH=end', 'Convergence lambda2 kH=0', 'Convergence lambda2 kH=5', 'Convergence lambda2 kH=10', 'Convergence lambda2 kH=20', 'Convergence lambda2 kH=40', 'Convergence lambda2 kH=80', 'Convergence lambda2 kH=end'])
    df.to_csv('./results/spss/dataprep_attaonly_caseII.csv', index=False)

