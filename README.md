# STEALTH: Don't Lie to Me: Avoiding Malicious Explanations with Stealth


## Overview

- `datasets` folder contains all datasets (raw and processed) and the pre-processing script used in this paper.
- `slack` folder contains the necessary files for the lying model from [Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods by Slack et. al.](https://dl.acm.org/doi/pdf/10.1145/3375627.3375830).
- `baselines` folder contains all the baselines used in the paper.  
    - `FairSMOTE` contains [Fair-SMOTE](https://arxiv.org/abs/2105.12195), which rebalances the protected attributes by generating additional
        synthetic training data. In Fair-SMOTE, `Generate_Samples.py` is used to generate synthetic data.
    - `MAAT` contains [MAAT](https://dl.acm.org/doi/pdf/10.1145/3540250.3549093), another fairness benchmark used,
        which combines models optimized for different objectives: fairness and ML performance.
    - `xFAIR` contains [FairMask](https://arxiv.org/pdf/2110.01109.pdf), another fairness benchmark used, which uses extrapolation models to relabel protected attributes later seen in testing data or deployment
        time. Their approach aims to offset the biased predictions of the classification model via rebalancing the distribution of protected attributes.
- `metrics`
    - `Measure.py` contains functions to compute performance measures and fairness measures.
    - `compareRanks.py` computes the Jaccard coefficient scores for the baseline and surrogate LIME rankings.
    - `medians.py` contains functions to compute the medians of performance measures and fairness measures from all runs. It helps consolidate.
    - `sk_setup.py` rearranges the output into the proper format for sk.py, and those files print to the "sk_data" folder.
    - `sk.py` contains Scott-Knott procedure.
- `output` folder is an example of the experiment output.
    - every dataset run will return the metrics and LIME rankings
- `STEALTH.py` runs on interation of the experiment.
- `cols.py` runs the structure for our bi-clustering implementation.
    


## Dataset Description

1. `Adult`: Adult Income dataset [available online](http://mlr.cs.umass.edu/ml/datasets/Adult)

2. `Bank`: Bank Marketing dataset [available online](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

3. `Communities`: Communities and Crime dataset [available online](http://www.ics.uci.edu/mlearn/ML-Repository)

4. `COMPAS`: ProPublica Recidivism dataset [available online](https://github.com/propublica/compas-analysis)

5. `Default`: Default payment dataset [available online](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

6. `Diabetes`: Pima Indians diabetes database [available online](https://kaggle.com/uciml/pima-indians-diabetes-database)

7. `German`: German Credit dataset [available online](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)

8. `Heart`: Cleveland Heart Disease dataset [available online](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

9. `MEPS`: Medical Expenditure Panel Survey dataset [available online](https://meps.ahrq.gov/mepsweb/)

10. `Student`: Student achievement dataset [available online](https://archive.ics.uci.edu/ml/datasets/Student+Performance)


To automate the experiment, we have processed the data into *`_p.csv` files using preprocessing.py to drop invalid data, unify the name of dependent variables, 
and selected relevant features. To replicate our experiment, please use the processed data.

## Experiment

To answer RQs presented in our paper, one can easily replicate our results:

RQ1: Does our method prevent lying?

RQ2: Does the surrogate model perform as well as the original model?

RQ3: How does STEALTH compare against other bias mitigation algorithms?


- Run `STEALTH.py` and set your run number in the terminal (see `STEALTH.sh` for an example). In the paper, we repeated the experiment for 20 runs using the same random seed for all models. We ran each iteration with HPC cluster computing rather than consecutively in a loop. However, the adjustment is an easy change. 
- Each run will have the RF baseline, Slack baseline, RF surrogate, and Slack surrogate performance, fairness, and LIME results for each dataset and protected attribute(s). It will also include the fairness `baselines`, and for the edge cases that aren't applicable, an error will print with some details about the problem and which dataset and model it occured with.  
- The output will be two files: (1) performance and fairness metric calculations 'datasetname + _metrics + run number.csv' and (2)  LIME ranking results 'datasetname + _LIME + run number.csv'
    - NOTE: Since LIME is used, the experiments can take up to 15+ hours for the larger datasets (Adult, Default) to complete, depending on the seed and computing power. However as each dataset completes, the results will output.
- Then these can be used to calculate median results with `medians.py` and jaccard coefficient scores comparing between the models with `compareRanks.py`. 
- The `sk_setup.py` will split the output for each metrics to run `sk.py`. `sk.py` will return a csv file with the Scott-Knott scores, and print to the terminal a graph representation. 
- The graph figures were made in Google sheet with minor spreadsheet manipulation. 

For more information, contact lalvare@ncsu.edu with STEALTH github in the subject line.