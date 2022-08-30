MATLAB was used for generating various plots and can also be used for data analysis.

## Plot Tasks in the Capability Hypercube `plot_tasks.m`

This script will plot the set of tasks for allocation in the capability hypercube along with the human's and robot's capabilities. The following lines of code can be edited:
* _5_ The .mat file for tasks to plot.
* _8_ Whether to show an animation of each task appearing in the capability hypercube. Set to 0 for no animation and 1 for animation.
* _15_ The number of tasks N for plotting.
* _51_ The amount of time to pause between each new task appearing in the capability hypersube. This line is only applicable if line _8_ is set to 1.


## Plot Allocations and Outcomes of Tasks in the Capability Hypercube for ATTA and Random Methods `plot_allocations_outcomes.m`

This script will plot which agent was allocated which task and the outcomes of those allocations for the ATTA and Random task allocation methods. The following lines of code can be edited:
* _12-15_ The .mat file with the allocations and outcomes to plot.
* _38, 57_ To change plot titles to match the task allocation method that is being plotted.
* _34, 35, 54_ To see the plot legend, uncomment these lines. To hide the plot legend, comment these lines.

In our paper, we plotted the allocations and outcomes from `code/results/atta/atta_caseI_eta50_3.mat` for ATTA case I, `code/results/atta/atta_caseII_eta50_3.mat` for ATTA case II, and `code/results/random/random_caseI_eta50_3.mat` for Random case I in Figures 3 and 4. 


## Plot Allocations and Outcomes of Tasks in the Capability Hypercube for Tsarouchi et al.'s Method `plot_allocations_outcomes_tsarouchi.m`

This script will plot which agent was allocated which task and the outcomes of those allocations for Tsarouchi et al.'s task allocation method. The following lines of code can be edited:
* _12-14_ The .mat file with the allocations and outcomes to plot.
* _15-16_ The offset used in Tsarouchi et al.'s method for each capability dimension. In case I, these lines should be set to 0.0. In case II, these lines are the offset used in the methods Tsarouchi et al. (+0.1) or Tsarouchi et al. (-0.1), either 0.1 or -0.1.
* _41, 62_ To change plot titles to match the task allocation method that is being plotted.
* _55_ In case II, uncomment this line to plot the inaccurately known human capabilities. Comment this line if plotting allocations and outcomes for case I.
* _37, 38, 59_ To see the plot legend, uncomment these lines. To hide the plot legend, comment these lines.

In our paper, we plotted the allocations and outcomes from `code/results/tsarouchi/tsarouchi_caseI_eta50_3.mat` for Tsarouchi et al. case I and `code/results/tsarouchi/tsarouchi_caseIIpos10_eta50_3.mat` for Tsarouchi et al. case II in Figures 3 and 4.


## Plot Progression of Capabilities Belief Distribution `plot_capabilities_update.m`

This script will plot the progression of the lower and upper bounds for an agent's capabilities belief distribution. The following lines of code can be edited:
* _7_ The .mat file with the ATTA case II results to plot.
* _60_ To see the plot legend, uncomment these lines. To hide the plot legend, comment these lines.

In our paper, we plotted the lower and upper bound progression from `code/results/atta/atta_caseII_eta50_3.mat` for ATTA case II in Figure 5.


## Plot Evolution in Trust `plot_trust_evolution.m`

This script will plot the evolution in trust across the capability hypercube after the human has executed k^H number of tasks. This script will plot trust across the capability hypercube for 0, 5, 10, 20, 40, 100, and end k^H number of tasks. End is the last task number executed by the human. Some code blocks may need to be commented out if the human was not allocated that many tasks. Use the 3D rotate tool on the generated plot to change the view. The following line of code can be edited:
* _3_ The .mat file with the ATTA case II results to plot.

In our paper, we plotted the trust evolution from `code/results/atta/atta_caseII_eta50_3.mat` for ATTA case II in Figure 6.
