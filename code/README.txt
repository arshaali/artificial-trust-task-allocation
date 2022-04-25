atta_caseII.py ###explore etas in here
Line 122 for \eta value
Line 129 for number of iterations
Line 141 learning rate
Line 142 weight decay
Line 146 loss tolerance to stop optimizer early
Line 149 report period
Line 163 nbins
Line 184 num tasks
220-223 human capabilities
228-231 robot capabilities
240 path to .mat tasks to allocate
307 task reward
309-310 human and robot cost
332 alpha tolerance
606 difference between current loss and loss 200 iterations ago at which to stop early
621 file name to save results to

random_caseI.py
19 for \eta value
Line 26 for number of iterations
Line 34 num tasks
62-71 human and robot capabilities
79 file to load tasks
116 task reward
118-119 human and robot cost
152 file name to save results to

roundrobin_caseI.py
19 eta 
Line 26 for number of iterations
Line 34 num tasks
62-71 human and robot capabilities
79 file to load tasks
117 task reward
119-120 human and robot cost
154 file name to save results to


tmmc_caseI_caseII.py
25 eta
Line 32 for number of iterations
Line 39 num tasks
67-76 human and robot capabilities
83-84 inaccurate offset. This is 0 for case I. something else for case II.
89 file to load tasks
107 task reward
109-110 human and robot cost
197 file name to save results to

atta_caseI.py ###explore costs in here
Line 122 for \eta value
Line 129 for number of iterations
Line 141 num tasks
175-178 human capabilities
183-186 robot capabilities
196 path to .mat tasks to allocate
222 task reward
224-225 human and robot cost
237 alpha tolerance
314 file name to save results to

atta_triad_caseI.py
Line 122 for \eta value
Line 129 for number of iterations
Line 143 num tasks
185-192 human capabilities
197-200 robot capabilities
211 path to .mat tasks to allocate
241 task reward
243-245 human and robot cost
261 alpha tolerance
424 file name to save results to


atta_triad_caseII.py
122 eta
129 number of iterations
143 learning rate
144 weight decay
Line 149 loss tolerance to stop optimizer early
Line 152 report period
Line 180 nbins
Line 196 num tasks
242-249 human capabilities
254-257 robot capabilities
268 path to .mat tasks to allocate
356 task reward
358-360 human and robot cost
385 alpha tolerance
756,946 difference between current loss and loss 200 iterations ago at which to stop early
961 file name to save results to



generate_uniformdist_tasks.py
11 number of iterations
13 num tasks
15 flag to save tasks
35 file name to save tasks to

generate_normaldist_tasks.py
13 number of iterations
15 num tasks
17 flag to save tasks
21-30 human and robot capabilities
38-39 sigma
75 file name to save tasks to

plot_normaldist_tasks.py
8 num tasks
11-18 human and robot capabilities
27-28 sigma should be same as in generate_normaldist_tasks.py
31 file to load tasks from


significance_prep_caseI.py
8 number of iterations
16 atta_caseI data files
38 tmmc_caseI data files
59 random data files
84 file name to save to

significance_prep_caseII.py
8 number of iterations
16 atta_caseI data files
38 tmmc_caseI data files
62 file name to save to

significance_prep_eta.py
8 number of iterations
16 atta_caseII_eta50 data files
75 atta_caseII_eta500 data files
134 atta_caseII_eta5 data files
209 file name to save to


explore_atta_results.py
8 number of iterations
16 atta_caseII files
95 file name to save to
Note: may have to change/remove some lines from 74-88 if the human was not allocated enough tasks for the calculation at that convergence offset




#check the line numbers are correct
#which beta value do we use when computing trust? I guess it doesnt matter because in the end, beta is > -50 and beta is not used in calculating trust then.
#TODO: 
#consider that the robot does not know its own capabilities. It should use the same approach to figure it out.

bin_centers is used to compute trust approximation.

#OBSERVATIONS/POINTS
#obs_prob_idxs start at 0 and go to 9
#how will it know what lower and upper bounds to converge to?
#    answer: the calculation of the performance is not based on the trust estimate based on current lower and upper bounds.
#        It is based on trust computed from the actual lower and upper bounds


To call the file TA_RobotTrustModel_2Dim.py, from the /code folder, type "python ./TA_RobotTrustModel_2Dim.py <task allocation method>", where <task allocation method> cab be replaced by "atta", "rr", or "rand" (without the quotation marks).

Lines 222-225 can be changed to change the human's capabilities. Lines 230-233 can be changed to change the robot's capabilities.

Line 183 can be changed to the number of tasks you want to allocate. If using a previously generated set of tasks, the number of tasks to allocate cannot be greater than the number of tasks that were previously generated. Otherwise, a new set of tasks should be generated.

If you would like to generate a new set of tasks, change the generatingNew flag in line 239 to True. If you would like to show the distribution and histogram of tasks figures, change the showing_fig flag in Line 258 to True. If you would like to save the distribution and histogram of tasks figures, change the saving_fig flag in Line 257 to True and the path and file names to save to in Line 302 and 317. If you would like to save these newly generated tasks, change the saving_new_tasks flag in Line 259 to True and the path and file name to save to in Line 322. The saving_fig, showing_fig, and saving_new_tasks flags will only take affect when generatingNew = True.

If you would like to load a previously generated set of tasks, make sure the generatingNew flag in Line 239 is False and the correct path to load from is in Line 327. 

If you want to change the task reward and agent cost functions for round-robin, change Lines 371-374.

To change the path and file for the results of running round-robin, change Line 404.

If you want to change the task reward and agent cost functions for random, change Lines 441-444.

To change the path and file for the results of running random, change Line 475.

If you want to change the task reward and agent cost functions for ATTA, change Lines 550-553.

To change the path and file for the results of running ATTA, change Line 830.

To change the \alpha tolerance in ATTA for assigning the task to the agent with fewer tasks already assigned to it, change Line 573.
