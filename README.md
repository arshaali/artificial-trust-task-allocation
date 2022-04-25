# Artificial Trust-Based Task Allocation (ATTA)

Dataset and software for the paper "Heterogeneous Human-Robot Task Allocation Based on Artificial Trust".

## Dataset

There are 10 sets of N=500 tasks for allocation found in `code/results/tasks`.


## Software

### Dependencies

All implementations were tested with Python 3.8.3 and PyTorch v1.7.1.
The following packages are needed (please install with `python3 -m pip install --user <package name>`):

* `numpy`
* `torch`
* `pickle`
* `scipy`
* `sklearn`
* `random`
* `sys`
* `matplotlib`
* `math`
* `time`
* `csv`
* `pandas`

### Task Allocation Methods Implementation

The ATTA method and code follows these steps in `code/atta_caseI.py` and `code/atta_caseII.py`:

* fabricate capabilities for both the human and robot
* load a .mat file with N tasks for allocation
* for each task:
  * compute the robot's trust in the human and robot's self-trust
  * compute the robot's and human's expected total reward
  * allocate the task to either the human or the robot
  * observe the outcome of the task as either a success or a failure
  * if the task was allocated to the human, update the human's capabilities belief distribution lower and upper bounds. This step is only done in `code/atta_caseII.py`.

There are two other task allocation methods evaluated in the paper. The random task allocation method and code follows these steps in `code/random_caseI.py`:

* fabricate capabilities for both the human and robot
* load a .mat file with N tasks for allocation
* for each task:
  * randomly allocate the task to either the human or the robot
  * observe the outcome of the task as either a success or a failure

Tsarouchi et al.'s task allocation method was implemented following the paper "On a human-robot collaboration in an assembly cell. International Journal of Computer Integrated Manufacturing, 30(6), 580-589." (Tsarouchi, P., Matthaiakis, A. S., Makris, S., & Chryssolouris, G., 2017). DOI: https://doi.org/10.1080/0951192X.2016.1187297. The implementation of Tsarouchi et al.'s task allocation method and code follows these steps in `code/tsarouchi_caseI.py` and `code/tsarouchi_caseII.py`:

* fabricate capabilities for both the human and robot
* load a .mat file with N tasks for allocation
* for each task:
  * if only one agent is capable of the task, allocate the task to that agent
  * if neither agent is capable of the task, discard the task
  * if both agents are capable of the task, allocate the task to the agent with the lower cost
  * observe the outcome of the task as either a success or a failure

## Use Instructions


### Task Allocation for Case I and Case II

1. Run `python generate_normaldist_tasks.py` from the `code` directory. This will generate .mat files with tasks for allocation in `code/results/tasks`.

2. Run the appropriate task allocation method(s) from the `code` directory. This will generate .mat files with the results of the task allocation in the respective directories of `code/results/atta`, `code/results/random`, and `code/results/tsarouchi`. The choices are:
  * `python atta_caseI.py`
  * `python atta_caseII.py`
  * `python random_caseI.py`
  * `python tsarouchi_caseI.py`
  * `python tsarouchi_caseII.py`

3. Generate .csv files for data analysis. This will prepare the data for statistical testing and analysis.
  * For case I, run `python significance_prep_caseI.py` which will generate `code/results/spss/dataprep_caseI.csv` and run `python other_data_caseI.py` which will generate `code/results/spss/other_data_caseI.csv`.
  * For case II, run `python significance_prep_caseII.py` which will generate `code/results/spss/dataprep_caseII_pos10.csv` and `code/results/spss/dataprep_caseII_neg10.csv` and run `python other_data_caseII.py` which will generate `code/results/spss/other_data_caseII_pos10.csv` and `code/results/spss/other_data_caseII_neg10.csv`.
  * For ATTA in case II only, run `python explore_atta_results.py` which will generate `code/results/spss/dataprep_attaonly_caseII.csv`.

4. Explore the results and data using IBM SPSS 26 and MATLAB.


## Data and Results

The results from each task allocation method for 10 simulations are in the respective directories of `code/results/atta`, `code/results/random`, and `code/results/tsarouchi`. 

The results in `code/results/spss` are from significance testing using IBM SPSS 26.

The results can be further explored in MATLAB using the scripts in `code/results/matlab_scripts`.


## Paper Source Files and Figures

The `paper` directory also contains the LaTeX source files for the paper.
Paper figures are in .svg format in `paper/SVG figures` and in .pdf format in `paper/PDF figures`.


## Misc

There are other README.txt files in various directories that provide further information on how to use the code.

