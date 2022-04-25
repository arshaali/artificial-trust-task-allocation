plot_tasks.m
To plot the human's capabilities, robot's capabilities, and the set of tasks for allocation. 
* 5 .mat file for tasks to plot
* 8 whether to show an animation of each task appearing. set to 0 for no animation and 1 for animation
* 15 number of tasks for plotting
* 51 the amount of time to pause between each new task appearing. only applicable if line 8 is set to 1.


plot_allocations_outcomes.m
To plot which agent got allocated which task and to plot the outcomes of those allocations for the ATTA and Random methods.
* 12-15 the .mat file with the allocations and outcomes to plot
* 38,57 change title of plots to match the task allocation method that is being plotted
* 34,35,54 uncomment this line if wanting to see the legend on the plot


plot_allocations_outcomes_tsarouchi.m
To plot which agent got allocated which task and to plot the outcomes of those allocations for Tsarouchi et al.'s method.
* 12-14 the .mat file with the allocations and outcomes to plot
* 15-16 the offset used in Tsarouchi et al. for each capability dimension. In case I, these lines should be set to 0.0. In case II, these lines are the offset used in Tsarouchi et al., either 0.1 or -0.1.
* 41,62 change title of plots to match the task allocation method that is being plotted
* 55 uncomment this line if plotting Tsarouchi et al.'s method in case II to show the inaccurately known human capabilities
* 37,38,59 uncomment this line if wanting to see the legend on the plot


plot_capabilities_update.m
To plot the progression of the lower and upper bounds for an agent's capabilities belief distribution
* 7 the .mat file with the ATTA case II results to plot
* 60 uncomment this line if wanting to see the legend


plot_trust_evolution.m
To plot the evolution in trust across the capability hypercube after the human has executed k^H number of tasks. 
Will automatically plot for 0, 5, 10, 20, 40, 100, and end. End is the last task executed by the human. May need to comment some out if the human was not allocated that many tasks.
* 3 the .mat file with the ATTA case II results to plot
use 3D rotate tool on plot to change the view


get_legends.m (not included in public repo)
some ugly plotting to get the legends for the figures


plot_uniformdist_tasks.m (not included in public repo)
To create a histogram of task requirements for each capability dimension. Only to visually see how uniformly distributed tasks are when the tasks are generated from a uniform distribution.
* 4 .mat file for uniform set of tasks to plot
* 6 number of tasks for plotting
