#imports
import scipy.io as sio
import csv
import pandas as pd 
import numpy as np


num_iter = 10 #number of times the strategies were run

if __name__ == "__main__":
    
    dataprep = []

    for iter in range(0, num_iter, 1):

        load_file_name = "./results/atta/atta_caseI_eta50_" + str(iter) + ".mat"
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


        team_success_rate_atta = 100 * ((human_successes_from_atta.shape[1] + robot_successes_from_atta.shape[1])/total_num_tasks_from_atta)
        human_success_rate_atta = 100 * ((human_successes_from_atta.shape[1])/human_num_tasks_from_atta)
        robot_success_rate_atta = 100 * ((robot_successes_from_atta.shape[1])/robot_num_tasks_from_atta)
        team_total_reward_atta = 100 * (total_reward_from_atta/max_total_reward_from_atta)
        perct_human_tasks_atta = 100 * (human_num_tasks_from_atta/total_num_tasks_from_atta)
        perct_robot_tasks_atta = 100 * (robot_num_tasks_from_atta/total_num_tasks_from_atta)


        load_file_name = "./results/tsarouchi/tsarouchi_caseI_eta50_" + str(iter) + ".mat"
        output_tsarouchi_mat = sio.loadmat(load_file_name)

        human_successes_from_tsarouchi = output_tsarouchi_mat["human_successes"] #tasks assigned to the human that were successes
        human_failures_from_tsarouchi = output_tsarouchi_mat["human_failures"] #tasks assigned to the human that were failures
        robot_successes_from_tsarouchi = output_tsarouchi_mat["robot_successes"] #tasks assigned to the robot that were successes
        robot_failures_from_tsarouchi = output_tsarouchi_mat["robot_failures"] #tasks assigned to the robot that were failures
        total_reward_from_tsarouchi = output_tsarouchi_mat["total_reward"][0][0]
        max_total_reward_from_tsarouchi = output_tsarouchi_mat["max_total_reward"][0][0]
        human_num_tasks_from_tsarouchi = output_tsarouchi_mat["human_num_tasks"][0][0]
        robot_num_tasks_from_tsarouchi = output_tsarouchi_mat["robot_num_tasks"][0][0]
        total_num_tasks_from_tsarouchi = output_tsarouchi_mat["total_num_tasks"][0][0]

        team_success_rate_tsarouchi = 100 * ((human_successes_from_tsarouchi.shape[1] + robot_successes_from_tsarouchi.shape[1])/total_num_tasks_from_tsarouchi)
        human_success_rate_tsarouchi = 100 * ((human_successes_from_tsarouchi.shape[1])/human_num_tasks_from_tsarouchi)
        robot_success_rate_tsarouchi = 100 * ((robot_successes_from_tsarouchi.shape[1])/robot_num_tasks_from_tsarouchi)
        team_total_reward_tsarouchi = 100 * (total_reward_from_tsarouchi/max_total_reward_from_tsarouchi)
        perct_human_tasks_tsarouchi = 100 * (human_num_tasks_from_tsarouchi/total_num_tasks_from_tsarouchi)
        perct_robot_tasks_tsarouchi = 100 * (robot_num_tasks_from_tsarouchi/total_num_tasks_from_tsarouchi)


        load_file_name = "./results/random/random_caseI_eta50_" + str(iter) + ".mat"
        output_random_mat = sio.loadmat(load_file_name)

        human_successes_from_random = output_random_mat["human_successes"] #tasks assigned to the human that were successes
        human_failures_from_random = output_random_mat["human_failures"] #tasks assigned to the human that were failures
        robot_successes_from_random = output_random_mat["robot_successes"] #tasks assigned to the robot that were successes
        robot_failures_from_random = output_random_mat["robot_failures"] #tasks assigned to the robot that were failures
        total_reward_from_random = output_random_mat["total_reward"][0][0]
        max_total_reward_from_random = output_random_mat["max_total_reward"][0][0]
        human_num_tasks_from_random = output_random_mat["human_num_tasks"][0][0]
        robot_num_tasks_from_random = output_random_mat["robot_num_tasks"][0][0]
        total_num_tasks_from_random = output_random_mat["total_num_tasks"][0][0]

        team_success_rate_random = 100 * ((human_successes_from_random.shape[1] + robot_successes_from_random.shape[1])/total_num_tasks_from_random)
        human_success_rate_random = 100 * ((human_successes_from_random.shape[1])/human_num_tasks_from_random)
        robot_success_rate_random = 100 * ((robot_successes_from_random.shape[1])/robot_num_tasks_from_random)
        team_total_reward_random = 100 * (total_reward_from_random/max_total_reward_from_random)
        perct_human_tasks_random = 100 * (human_num_tasks_from_random/total_num_tasks_from_random)
        perct_robot_tasks_random = 100 * (robot_num_tasks_from_random/total_num_tasks_from_random)


        row = [team_success_rate_atta, human_success_rate_atta, robot_success_rate_atta, team_total_reward_atta, perct_human_tasks_atta, perct_robot_tasks_atta, team_success_rate_tsarouchi, human_success_rate_tsarouchi, robot_success_rate_tsarouchi, team_total_reward_tsarouchi, perct_human_tasks_tsarouchi, perct_robot_tasks_tsarouchi, team_success_rate_random, human_success_rate_random, robot_success_rate_random, team_total_reward_random, perct_human_tasks_random, perct_robot_tasks_random]
        dataprep.append(row)

    df = pd.DataFrame(dataprep, columns=['ATTA Team Perf', 'ATTA Human Perf', 'ATTA Robot Perf', 'ATTA Team Total Reward', 'ATTA Perct Human Tasks', 'ATTA Perct Robot Tasks', 'Tsarouchi Team Perf', 'Tsarouchi Human Perf', 'Tsarouchi Robot Perf', 'Tsarouchi Team Total Reward', 'Tsarouchi Perct Human Tasks', 'Tsarouchi Perct Robot Tasks', 'Random Team Perf', 'Random Human Perf', 'Random Robot Perf', 'Random Team Total Reward', 'Random Perct Human Tasks', 'Random Perct Robot Tasks'])
    df.to_csv('./results/spss/dataprep_caseI.csv', index=False)

