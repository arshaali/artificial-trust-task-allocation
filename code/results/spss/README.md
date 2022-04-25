All statistical testing was done using the one-tailed Wilcoxon signed-rank test which compares medians.

## Case I Results and Statistical Testing

The results for statistical testing from the ATTA, random, and Tsarouchi et al. task allocation methods in case I are in `dataprep_caseI.csv`. Statistical testing was done on the metrics of team performance, human performance, robot performance, and team total reward. 

Other results on the number of successes and failures are in `other_data_caseI.csv`. For convenience, an Excel version is also provided in `other_data_caseI.xlsx`.

## Case II Results and Statistical Testing

The results for statistical testing from the ATTA and Tsarouchi et al. (+0.1) task allocation methods in case II are in `dataprep_caseII_pos10.csv`. The results for statistical testing from the ATTA and Tsarouchi et al. (-0.1) task allocation methods in case II are in `dataprep_caseII_neg10.csv`.Statistical testing was done on the metrics of team performance, human performance, robot performance, and team total reward. 

More results from ATTA for case II, including covergence offsets, are in `dataprep_attaonly_caseII.csv`.

Other results on the number of successes and failures are in `other_data_caseII_pos10.csv` and `other_data_caseII_neg10.csv`. For convenience, Excel versions are also provided in `other_data_caseII_pos10.xlsx` and `other_data_caseII_neg10.xlsx`.

## How to use IBM SPSS 26 for Descriptives (Statistics) and Statistical Testing
1. Import the .csv file into SPSS. This will result in corresponding .sav files. The choices are:
  * `dataprep_caseI.csv`
  * `dataprep_caseII_pos10.csv`
  * `dataprep_caseII_neg10.csv`
  * `dataprep_attaonly_caseII.csv`
You can check that all variables are set to "ordinal" in the "variable view" tab.

2. To generate descriptives (median, mean, standard deviation, etc.), click on `Analyze > Descriptive Statistics > Explore`. Move the variables you want to generate descriptives for to the "Dependent List". To generate statistics only and no plots, move the radio button to "Statistics". Refer to `spss_explore.png` for what this should look like. Click "OK" to generate the descriptives. 
This is what was done for `dataprep_attaonly_caseII.csv`. The SPSS output log is `data_attaonly_caseII.spv` and the web version is `data_attaonly_caseII.htm`.
Step #2 (and step #3) was also done for the other 3 files listed in step #1.

3. To perform the Wilcoxon signed-rank test, click on `Analyze > Nonparametric Tests > Legacy Dialogs > 2 Related Samples`. Move the dependent variables pairs to test into "Test Pairs". "Wilcoxon" should be checked with other test types unchecked by default. Click on "Exact" and move the radio button to "Exact" with the "Time limit per test" checked and set to 5 minutes. Refer to `spss_exact.png` for what this should look like. Click "Continue" to return to the previous screen and then click "OK" to generate the results of the statistical tests. Refer to `spss_wilcoxon.png` for what this should look like.
This produced the following SPSS output logs and web versions:
  * `data_caseI.spv` and `data_caseI.htm`
  * `data_caseII_pos10.spv` and `data_caseII_pos10.htm`
  * `data_caseII_neg10.spv` and `data_caseII_neg10.htm`

4. The easiest way to explore the results is to double click the .htm file from your file explorer to open it in a web browser. Click on "Descriptives" to see various statistics. Click on "Test Statistics" to see the p-values for the one-tailed test. For significant p-values, click on "Ranks" to determine which dependent variable is higher or lower than the other.

## Misc

The `other_data_caseI.csv`, `other_data_caseII_pos10.csv`, and `other_data_caseII_neg10.csv` files were used to support the statements made in the Discussion section of the paper. These were not imported into SPSS.
