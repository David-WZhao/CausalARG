# Causal-ARG

This is the code of the Causal-ARG in our manuscript
### Requirement
- python 3.8
- torch == 1.12.1
- scikit-learn == 1.2.2
- numpy == 1.24.2
- pandas == 1.3.4

### Data
The process of collecting data is described in our manuscript and metadata can be accessed upon request.

### How to run the code?
1. Data preprocessing: "arg_v5.fasta" file is the original data set file, "fasta_process.ipynb" file is used on the original data set file to get the processed dataset.
Run "data_divide.py" to produce splitted dataset.

2. Run the prediction model: Put the "data_loader.py", "modules.py", "run.py", "utils.py" and directory "data" in the same directory, and input the running command in the following format:

python --device []    --batch_size []  --K  []  --X-dim  []   --G-dim  [] --z1_dim [] --z2_dim []
example: python run_record_loss.py --device "gpu" --batch_size 32 --K 5 --X_dim 64 --G_dim 64 --z1_dim 64 --z2_dim 64
