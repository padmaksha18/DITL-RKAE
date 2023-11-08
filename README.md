# DLSCA-AE

* The model is coded in "AE.py"
* Run it as below. Choose the hyperparams accordingly.
* python3 AE.py --code_size 9 --w_reg 0.001 --a_reg 0.2 --num_epochs 25 --max_gradient_norm 0.5 --learning_rate 0.001 --hidden_size 64
* TS_datasets.py has the file location for the train and test data. The DATA folder has the normal and anomaly files.
The details and results can be found in the paper.
