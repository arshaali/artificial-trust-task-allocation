These Python scripts were used to generate tasks for allocation, run the task allocation methods, and organize and prepare the results for statistical testing and analysis. Where lines of code can be edited, make sure they are consistent with the values used in the other scripts.

## Generate Tasks `generate_normaldist_tasks.py`

This script will generate sets of N tasks for allocation. The following lines of code can be edited:
* _13_ The number of sets/iterations of N tasks to generate.
* _15_ The number of tasks N to allocate. 
* _17_ A flag for whether to save the set of generated tasks. Set to True to save the tasks or False to not save the tasks.
* _21-30_ The fabricated values for the human's and robot's capabilities.
* _38-39_ The standard deviation (sigma) for each capability dimension distribution from which task requirements for that dimension will be sampled from.
* _75_ The file name to save tasks to.

In our paper, we generated 10 sets of N = 500 tasks.


## Plot Distributions and Histograms Tasks were Sampled From `plot_normaldist_tasks.py`

This script will plot the distribution for each capability dimension from which task requirements were sampled from. The histograms for sampled tasks are overlaid on the same plot. The tasks should have already been sampled from `generate_normaldist_tasks.py`. The following lines of code can be edited:
* _8_ The number of tasks N to allocate.
* _11-18_ The fabricated values for the human's and robot's capabilities.
* _27-28_ The standard deviation (sigma) for each capability dimension distribution from which task requirements for that dimension will be sampled from.
* _31_ The file name to load generated tasks from. This should match the file name saved to in line _75_ when generating the tasks. 

In our paper, we plotted the tasks generated in `code/results/tasks/TA_normaldist_tasks_3.mat`.


## Artificial Trust-Based Task Allocation (ATTA) Method Implementation `atta_caseI.py` and `atta_caseII.py`

These scripts will run the ATTA method implementation for case I and case II. The following lines of code can be edited:
atta_caseI.py
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

atta_caseII.py 
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


## Random Method Implementation `random_caseI.py`

These scripts will run the Random method implementation for case I (no difference in implementation for case II). The following lines of code can be edited:

random_caseI.py
19 for \eta value
Line 26 for number of iterations
Line 34 num tasks
62-71 human and robot capabilities
79 file to load tasks
116 task reward
118-119 human and robot cost
152 file name to save results to


## Tsarouchi et al.'s Method Implementation `tsarouchi_caseI.py` and `tsarouchi_caseII.py`

These scripts will run the ATTA method implementation for case I and case II. The following lines of code can be edited:

`tsarouchi_caseI.py` CHECK THESE LINES
25 eta
Line 32 for number of iterations
Line 39 num tasks
67-76 human and robot capabilities
83-84 inaccurate offset. This is 0 for case I. something else for case II.
89 file to load tasks
107 task reward
109-110 human and robot cost
197 file name to save results to



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



explore_atta_results.py
8 number of iterations
16 atta_caseII files
95 file name to save to
Note: may have to change/remove some lines from 74-88 if the human was not allocated enough tasks for the calculation at that convergence offset



#OBSERVATIONS/POINTS
#obs_prob_idxs start at 0 and go to 9
#how will it know what lower and upper bounds to converge to?
#    answer: the calculation of the performance is not based on the trust estimate based on current lower and upper bounds.
#        It is based on trust computed from the actual lower and upper bounds
