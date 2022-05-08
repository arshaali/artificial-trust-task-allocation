These Python scripts were used to generate tasks for allocation, run the task allocation methods, and organize and prepare the results for statistical testing and analysis. Where lines of code can be edited, make sure they are consistent with the values used in the other scripts. All scripts are run from the `code` directory by typing `python <file name>.py` in the terminal.

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

These scripts will run the ATTA method implementation for case I and case II. For ATTA in case I, the following lines of code can be edited:
* _122_ The eta value for uncertainty in task outcomes.
* _129_ The number of sets/iterations of N tasks to allocate.
* _141_ The number of tasks N to allocate.
* _175-178_ The human's capabilities.
* _183-186_ The robot's capabilities.
* _196_ The .mat file with tasks to allocate.
* _222_ The task reward equation.
* _224-225_ The human's and robot's cost equation.
* _237_ The alpha tolerance for when expected total rewards should be considered approximately the same.
* _314_ The file name to save the results of the task allocations and outcomes to.

For ATTA in case II, the following lines of code can be edited:
* _122_ The eta value for uncertainty in task outcomes.
* _129_ The number of sets/iterations of N tasks to allocate.
* _141_ The learning rate of the Pytorch Adam optimizer.
* _142_ The weight decay of the Pytorch Adam optimizer.
* _146_ The loss tolerance to stop the optimizer early for when the loss is less than this value.
* _149_ The report period for how often (number of optimizer iterations) to print lower and upper bounds and the loss to the terminal.
* _163_ The number of bins nbins to discretize each capability dimension with. 
* _184_ The number of tasks N to allocate.
* _220-223_ The human's capabilities.
* _228-231_ The robot's capabilities.
* _240_ The .mat file with tasks to allocate.
* _307_ The task reward equation.
* _309-310_ The human's and robot's cost equation.
* _332_ The alpha tolerance for when expected total rewards should be considered approximately the same.
* _606_ When the difference between the current loss and loss report_period iterations ago falls within this value, the optimizer can stop early since the loss is not decreasing.
* _621_ The file name to save the results of the task allocations and outcomes to.


## Random Method Implementation `random_caseI.py`

These scripts will run the Random method implementation for case I (no difference in implementation for case II). The following lines of code can be edited:
* _19_ The eta value for uncertainty in task outcomes.
* _26_ The number of sets/iterations of N tasks to allocate.
* _34_ The number of tasks N to allocate.
* _62-71_ The human's and robot's capabilities.
* _79_ The .mat file with tasks to allocate.
* _116_ The task reward equation.
* _118-119_ The human's and robot's cost equation.
* _152_ The file name to save the results of the task allocations and outcomes to.


## Tsarouchi et al.'s Method Implementation `tsarouchi_caseI.py` and `tsarouchi_caseII.py`

These scripts will run Tsarouchi et al.'s method implementation for case I and case II. For Tsarouchi et al.'s method in case I, the following lines of code can be edited:
* _25_ The eta value for uncertainty in task outcomes.
* _32_ The number of sets/iterations of N tasks to allocate.
* _39_ The number of tasks N to allocate.
* _73-82_ The human's and robot's capabilities.
* _91_ The .mat file with tasks to allocate.
* _109_ The task reward equation.
* _111-112_ The human's and robot's cost equation.
* _221_ The file name to save the results of the task allocations and outcomes to.

For Tsarouchi et al.'s method in case II, the following lines of code can be edited:
* _25_ The eta value for uncertainty in task outcomes.
* _32_ The number of sets/iterations of N tasks to allocate.
* _39_ The number of tasks N to allocate.
* _73-82_ The human's and robot's capabilities.
* _89-90_ The inaccurate offset for the human's capabilities for each capability dimension.
* _94_ The .mat file with tasks to allocate.
* _112_ The task reward equation.
* _114-115_ The human's and robot's cost equation.
* _224_ The file name to save the results of the task allocations and outcomes to.


## Prepare Case I Results for Significance Testing `significance_prep_caseI.py`

This script will organize and prepare the results from the ATTA, Random, and Tsarouchi et al. methods from case I for signifiance testing. The metrics prepared are team performance, human performance, robot performance, team total reward, percentage of tasks allocated to the human, and percentage of tasks allocated to the robot. The first four metrics are compared for significance in SPSS. Please see the README in `code/results/spss` for more information of significance testing. The following lines of code can be edited:
* _8_ The number of sets/iterations of N tasks to allocate.
* _16_ The .mat files for the results from the ATTA method for case I.
* _38_ The .mat files for the results from the Tsarouchi et al. method for case I.
* _59_ The .mat files for the results from the Random method for case I.
* _84_ The file name to save the prepared data and metrics to.


## Prepare Case II Results for Significance Testing `significance_prep_caseII.py`

This script will organize and prepare the results from the ATTA and Tsarouchi et al. methods from case II for signifiance testing. The metrics prepared are team performance, human performance, robot performance, team total reward, percentage of tasks allocated to the human, and percentage of tasks allocated to the robot. The first four metrics are compared for significance in SPSS. Please see the README in `code/results/spss` for more information of significance testing. The following lines of code can be edited:
* _8_ The number of sets/iterations of N tasks to allocate.
* _16_ The .mat files for the results from the ATTA method for case II.
* _38_ The .mat files for the results from the Tsarouchi et al. method for case II.
* _63_ The file name to save the prepared data and metrics to.


## Prepare ATTA Case II Results for SPSS Descriptives `explore_atta_results.py`

This script will organize and prepare the results from the ATTA method from case II for computing descriptives (statistics like median, mean, standard deviation, etc.) in SPSS. The metrics prepared are team performance, human performance, robot performance, team total reward, percentage of tasks allocated to the human, percentage of tasks allocated to the robot, and convergence offset data for both capability dimensions. Convergence offset data for each capability dimension is included for 0, 5, 10, 20, 40, 80, and end k^H number of tasks. End is the last task number executed by the human. The following lines of code can be edited: 
* _8_ The number of sets/iterations of N tasks to allocate.
* _16_ The .mat files for the results from the ATTA method for case II.
* _74-88_ Some lines for convergence offset may need to be altered or removed if the human was not allocated that many tasks.
* _95_ The file name to save the prepared data and metrics to.


## Prepare Other Case I Results for Analysis `other_data_caseI.py`

This script will organize and prepare additional results from the ATTA, Random, and Tsarouchi et al. methods from case I for observation and analysis. Other data metrics prepared include the number of task successes, number of task failures, number of tasks allocated to each agent, number of tasks discarded and how many to each agent in Tsarouchi et al.'s method implementation, number of human task successes, number of human task failures, number of robot task successes, number of robot task failures, number of failed and discarded tasks attributed to the human in Tsarouchi et al.'s method implementation, and number of failed and discarded tasks attributed to the robot in Tsarouchi et al.'s method implementation. The following lines of code can be edited:
* _8_ The number of sets/iterations of N tasks to allocate.
* _16_ The .mat files for the results from the ATTA method for case I.
* _37_ The .mat files for the results from the Tsarouchi et al. method for case I.
* _64_ The .mat files for the results from the Random method for case I.
* _102_ The file name to save the prepared data and metrics to.


## Prepare Other Case II Results for Analysis `other_data_caseII.py`

This script will organize and prepare additional results from the ATTA and Tsarouchi et al. methods from case II for observation and analysis. Other data metrics prepared include the number of task successes, number of task failures, number of tasks allocated to each agent, number of tasks discarded and how many to each agent in Tsarouchi et al.'s method implementation, number of human task successes, number of human task failures, number of robot task successes, number of robot task failures, number of failed and discarded tasks attributed to the human in Tsarouchi et al.'s method implementation, and number of failed and discarded tasks attributed to the robot in Tsarouchi et al.'s method implementation. The following lines of code can be edited:
* _8_ The number of sets/iterations of N tasks to allocate.
* _16_ The .mat files for the results from the ATTA method for case II.
* _37_ The .mat files for the results from the Tsarouchi et al. method for case II.
* _80_ The file name to save the prepared data and metrics to.
