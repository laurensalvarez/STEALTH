# STEALTH
To run the experiments, the main file is extraction.py and is called with "python3 extraction.py" in your terminal. There is also a bash script with the command titled extraction.sh

The output of extraction.py will be the metric scores and LIME explanations for each repeitition. Since LIME is used, the experiments can take up to 24 hours for the larger datasets (Adult, Default) to complete. However after each dataset completes, the results output into a file with the dataset name.

To get the results summarizations, compareRanks.py computes the jaccard coefficient, and sk.py computes the Scott Knott procedure. To run, sk.py the results have to be a specified format which can be created using sksetup.py.

For more information, contact me at lalvare@ncsu.edu with STEALTH github in the subject line. 
