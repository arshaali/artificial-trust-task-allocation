# Artificial Trust-Based Task Allocation (ATTA)

Dataset and software for task allocation based on artificial trust from the paper "Heterogeneous Human-Robot Task Allocation Based on Artificial Trust".

## Dataset



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

### Method Implementation

The ATTA method and code follows these steps:

* fabricate capabilities for both the human and robot
* fabricate N tasks with task requirements into a .mat file
* load the tasks with task requirements from the .mat file
* for each task:
  * compute the robot's trust in the human and robot's self-trust
  * compute the robot's and human's expected total reward
  * allocate the task to either the human or the robot
  * observe the outcome of the task as either a success or a failure
  * if the task was allocated to the human, update the human's capabilities belief distribution lower and upper bounds

## Use Instructions


### Task Allocation from Robots' Artificial Trust

Below is wrong and not updated.

1. Run `data/robotTrust_dataGen.m` on MATLAB (optional).
  (That will generate the sythentic data for the Task Allocation Artificial Trust Mode simulation. Here, line _23_ and _41_ can be changed to represent the value of _N_.)

2. Run `python TA_RobotTrustModel_2Dim.py` from the `code` directory.
  (That will generate the file `results/resultsRobotTrust_2Dim_TA.mat`.)

3. Explore the results file `results/resultsRobotTrust_2Dim_TA.mat` on MATLAB.




## Paper Figures





## Misc


