#imports
import scipy.io as sio
import csv
import pandas as pd 
import numpy as np


num_iter = 10 #number of times the strategies were run

if __name__ == "__main__":
    
    dataprep = []

    for iter in range(0, num_iter, 1):

        human_cost_for_failures_atta = 0 #sum of the cost incurred for the tasks the human failed on
        robot_cost_for_failures_atta = 0 #sum of the cost incurred for the tasks the robot failed on
        human_cost_for_failures_tsarouchi = 0 #sum of the cost incurred for the tasks the human failed on and were discarded to the human
        robot_cost_for_failures_tsarouchi = 0 #sum of the cost incurred for the tasks the robot failed on and were discarded to the robot
        

        load_file_name = "./results/atta/atta_caseII_eta50_" + str(iter) + ".mat"
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

        human_num_failures_from_atta = len(human_failures_from_atta[0]) #the number of tasks the human failed on
        for i in range(human_num_failures_from_atta):
            human_cost_for_failures_atta += (human_failures_from_atta[0][i] + human_failures_from_atta[1][i])/(3.0) #compute the human cost for every task the human failed on. This is the same human cost equation used during allocation.        
            
        robot_num_failures_from_atta = len(robot_failures_from_atta[0]) #the number of tasks the robot failed on
        for i in range(robot_num_failures_from_atta):
            robot_cost_for_failures_atta += (robot_failures_from_atta[0][i] + robot_failures_from_atta[1][i])/(8.0) #compute the robot cost for every task the robot failed on. This is the same robot cost equation used during allocation.


        load_file_name = "./results/tsarouchi/tsarouchi_caseIIpos10_eta50_" + str(iter) + ".mat"
        #load_file_name = "./results/tsarouchi/tsarouchi_caseIIneg10_eta50_" + str(iter) + ".mat"
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

        human_num_failures_from_tsarouchi = len(human_failures_from_tsarouchi[0]) #the number of tasks the human failed on
        for i in range(human_num_failures_from_tsarouchi):
            human_cost_for_failures_tsarouchi += (human_failures_from_tsarouchi[0][i] + human_failures_from_tsarouchi[1][i])/(3.0) #compute the human cost for every task the human failed on. This is the same human cost equation used during allocation.

        for i in range(discarded_num_humanCost_from_tsarouchi):
            human_cost_for_failures_tsarouchi += (0.65)/(3.0) #add the human cost for every task discarded to the human. 
            #Note: the task requirements for tasks discarded to the human and counted as failures were not recorded during the simulation. However, one \lambda_i for every task must be > 0.65 which is enough to support the claims made in the discussion.

        robot_num_failures_from_tsarouchi = len(robot_failures_from_tsarouchi[0]) #the number of tasks the robot failed on
        for i in range(robot_num_failures_from_tsarouchi):
            robot_cost_for_failures_tsarouchi += (robot_failures_from_tsarouchi[0][i] + robot_failures_from_tsarouchi[1][i])/(8.0) #compute the robot cost for every task the robot failed on. This is the same robot cost equation used during allocation.

        for i in range(discarded_num_robotCost_from_tsarouchi):
            robot_cost_for_failures_tsarouchi += (0.65)/(8.0) #add the robot cost for every task discarded to the robot. 
            #Note: the task requirements for tasks discarded to the robot and counted as failures were not recorded during the simulation. However, one \lambda_i for every task must be > 0.65.

        
        row = [num_successes_from_atta, num_failures_from_atta, 
        num_successes_from_tsarouchi, num_failures_from_tsarouchi,
        human_num_tasks_from_atta, robot_num_tasks_from_atta,
        human_num_tasks_from_tsarouchi, robot_num_tasks_from_tsarouchi, discarded_num_tasks_from_tsarouchi, discarded_num_humanCost_from_tsarouchi, discarded_num_robotCost_from_tsarouchi,     
        human_successes_from_atta.shape[1], human_failures_from_atta.shape[1], robot_successes_from_atta.shape[1], robot_failures_from_atta.shape[1],
        human_successes_from_tsarouchi.shape[1], human_failures_from_tsarouchi.shape[1], human_failures_from_tsarouchi.shape[1] + discarded_num_humanCost_from_tsarouchi, 
        robot_successes_from_tsarouchi.shape[1], robot_failures_from_tsarouchi.shape[1], robot_failures_from_tsarouchi.shape[1] + discarded_num_robotCost_from_tsarouchi,
        human_cost_for_failures_atta, robot_cost_for_failures_atta, human_cost_for_failures_tsarouchi, robot_cost_for_failures_tsarouchi]
        
        dataprep.append(row)

    df = pd.DataFrame(dataprep, columns=['ATTA Num Succ', 'ATTA Num Fail', 'Tsarouchi Num Succ', 'Tsarouchi Num Fail', 
    'ATTA H Num Tasks', 'ATTA R Num Tasks', 
    'Tsarouchi H Num Tasks', 'Tsarouchi R Num Tasks', 'Tsarouchi Discarded Num Tasks', 'Tsarouchi Discarded Num to H', 'Tsarouchi Discarded Num to R',
    'ATTA H Num Succ', 'ATTA H Num Fail', 'ATTA R Num Succ', 'ATTA R Num Fail',
    'Tsarouchi H Num Succ', 'Tsarouchi H Num Fail', 'Tsarouchi H Num Fail + Discarded', 'Tsarouchi R Num Succ', 'Tsarouchi R Num Fail', 'Tsarouchi R Num Fail + Discarded',
    'ATTA H Cost for Failures', 'ATTA R Cost for Failures', 'Tsarouchi MIN H Cost for Failures', 'Tsarouchi MIN R Cost for Failures'])
    df.to_csv('./results/spss/other_data_caseII_pos10.csv', index=False)
    #df.to_csv('./results/spss/other_data_caseII_neg10.csv', index=False)

