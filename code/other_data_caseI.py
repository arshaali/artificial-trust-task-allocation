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
        human_num_tasks_from_atta = output_atta_mat["human_num_tasks"][0][0]
        robot_num_tasks_from_atta = output_atta_mat["robot_num_tasks"][0][0]


        num_successes_from_atta = human_successes_from_atta.shape[1] + robot_successes_from_atta.shape[1]
        num_failures_from_atta = human_failures_from_atta.shape[1] + robot_failures_from_atta.shape[1]
        #human_num_tasks_from_atta
        #robot_num_tasks_from_atta
        #human_successes_from_atta.shape[1]
        #human_failures_from_atta.shape[1]
        #robot_successes_from_atta.shape[1]
        #robot_failures_from_atta.shape[1]


        load_file_name = "./results/tsarouchi/tsarouchi_caseI_eta50_" + str(iter) + ".mat"
        output_tsarouchi_mat = sio.loadmat(load_file_name)

        human_successes_from_tsarouchi = output_tsarouchi_mat["human_successes"] #tasks assigned to the human that were successes
        human_failures_from_tsarouchi = output_tsarouchi_mat["human_failures"] #tasks assigned to the human that were failures
        robot_successes_from_tsarouchi = output_tsarouchi_mat["robot_successes"] #tasks assigned to the robot that were successes
        robot_failures_from_tsarouchi = output_tsarouchi_mat["robot_failures"] #tasks assigned to the robot that were failures
        human_num_tasks_from_tsarouchi = output_tsarouchi_mat["human_num_tasks"][0][0]
        robot_num_tasks_from_tsarouchi = output_tsarouchi_mat["robot_num_tasks"][0][0]
        discarded_num_tasks_from_tsarouchi = output_tsarouchi_mat["discarded_num_tasks"][0][0]
        discarded_num_robotCost_from_tsarouchi = output_tsarouchi_mat["discarded_num_robotCost"][0][0]
        discarded_num_humanCost_from_tsarouchi = output_tsarouchi_mat["discarded_num_humanCost"][0][0]

        
        num_successes_from_tsarouchi = human_successes_from_tsarouchi.shape[1] + robot_successes_from_tsarouchi.shape[1]
        num_failures_from_tsarouchi = human_failures_from_tsarouchi.shape[1] + robot_failures_from_tsarouchi.shape[1] + discarded_num_tasks_from_tsarouchi
        #human_num_tasks_from_tsarouchi
        #robot_num_tasks_from_tsarouchi
        #discarded_num_tasks_from_tsarouchi
        #discarded_num_humanCost_from_tsarouchi
        #discarded_num_robotCost_from_tsarouchi
        #human_successes_from_tsarouchi.shape[1]
        #human_failures_from_tsarouchi.shape[1]
        #robot_successes_from_tsarouchi.shape[1]
        #robot_failures_from_tsarouchi.shape[1]

        
        load_file_name = "./results/random/random_caseI_eta50_" + str(iter) + ".mat"
        output_random_mat = sio.loadmat(load_file_name)

        human_successes_from_random = output_random_mat["human_successes"] #tasks assigned to the human that were successes
        human_failures_from_random = output_random_mat["human_failures"] #tasks assigned to the human that were failures
        robot_successes_from_random = output_random_mat["robot_successes"] #tasks assigned to the robot that were successes
        robot_failures_from_random = output_random_mat["robot_failures"] #tasks assigned to the robot that were failures
        human_num_tasks_from_random = output_random_mat["human_num_tasks"][0][0]
        robot_num_tasks_from_random = output_random_mat["robot_num_tasks"][0][0]


        num_successes_from_random = human_successes_from_random.shape[1] + robot_successes_from_random.shape[1]
        num_failures_from_random = human_failures_from_random.shape[1] + robot_failures_from_random.shape[1]
        #human_num_tasks_from_random
        #robot_num_tasks_from_random
        #human_successes_from_random.shape[1]
        #human_failures_from_random.shape[1]
        #robot_successes_from_random.shape[1]
        #obot_failures_from_random.shape[1]



        row = [num_successes_from_atta, num_failures_from_atta, num_successes_from_random, num_failures_from_random,
        num_successes_from_tsarouchi, num_failures_from_tsarouchi,
        human_num_tasks_from_atta, robot_num_tasks_from_atta, human_num_tasks_from_random, robot_num_tasks_from_random,
        human_num_tasks_from_tsarouchi, robot_num_tasks_from_tsarouchi, discarded_num_tasks_from_tsarouchi, discarded_num_humanCost_from_tsarouchi, discarded_num_robotCost_from_tsarouchi,     
        human_successes_from_atta.shape[1], human_failures_from_atta.shape[1], robot_successes_from_atta.shape[1], robot_failures_from_atta.shape[1],
        human_successes_from_random.shape[1], human_failures_from_random.shape[1], robot_successes_from_random.shape[1], robot_failures_from_random.shape[1],
        human_successes_from_tsarouchi.shape[1], human_failures_from_tsarouchi.shape[1], human_failures_from_tsarouchi.shape[1] + discarded_num_humanCost_from_tsarouchi, 
        robot_successes_from_tsarouchi.shape[1], robot_failures_from_tsarouchi.shape[1], robot_failures_from_tsarouchi.shape[1] + discarded_num_robotCost_from_tsarouchi]
        
        dataprep.append(row)

    df = pd.DataFrame(dataprep, columns=['ATTA Num Succ', 'ATTA Num Fail', 'Random Num Succ', 'Random Num Fail', 'Tsarouchi Num Succ', 'Tsarouchi Num Fail', 
    'ATTA H Num Tasks', 'ATTA R Num Tasks', 'Random H Num Tasks', 'Random R Num Tasks', 
    'Tsarouchi H Num Tasks', 'Tsarouchi R Num Tasks', 'Tsarouchi Discarded Num Tasks', 'Tsarouchi Discarded Num to H', 'Tsarouchi Discarded Num to R',
    'ATTA H Num Succ', 'ATTA H Num Fail', 'ATTA R Num Succ', 'ATTA R Num Fail', 'Random H Num Succ', 'Random H Num Fail', 'Random R Num Succ', 'Random R Num Fail',
    'Tsarouchi H Num Succ', 'Tsarouchi H Num Fail', 'Tsarouchi H Num Fail + Discarded', 'Tsarouchi R Num Succ', 'Tsarouchi R Num Fail', 'Tsarouchi R Num Fail + Discarded'])
    df.to_csv('./results/spss/other_data_caseI.csv', index=False)

