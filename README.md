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
To facilitate use, in addition to the source code (i.e. `.py` files)  for each module, we also provide `.ipynb` files which include the entire process of model training. Simply download the code to your local machine and execute `run.ipynb` each code block in order to proceed with the training.



In addtion, we also provide a Colab link(https://colab.research.google.com/drive/1lzLFccXoGSYej1iVb1Y3TGuGfvF-EDDi?usp=sharing) to use our pre-trained model online. Through this link, you can directly access the code we provide for calling our trained model.

You can also open the `triained-model.ipynb` to use our trained model.

# Introduction of Baselines
Baselines
To comprehensively evaluate the effectiveness of the proposed method, representative methods are selected as baselines for performance comparison, which are listed as follows.

## BestHit
This method is conducted by comparing the sample sequences with existing ARGs in CARD by applying the BLAST or DIAMOND, and the predicted properties are assigned to samples through applying a similarity cutoff. Note that BestHit can be used only for predicting antibiotic classes and resistance mechanisms of ARGs.
For detailed usage of the CARD database, please refer to the link: https://card.mcmaster.ca/

## DeepARG
This method is a deep learning-based model which is trained by taking the consistency distribution of homologies between sample sequences and all known ARGs as input features. Note that DeepARG can be used only for predicting antibiotic classes of ARGs.
For detailed usage of the DeepARG, please refer to the link: https://github.com/gaarangoa/deeparg

## HMD-ARG
This method extracts features from raw sequences through an end-to-end deep CNN-based framework for predicting properties of ARGs. Note that HMD-ARG is a multi-task model, which can be used for predicting all of three properties of ARGs.
For detailed usage of the HMD-ARG:, please refer to the link:http://www.cbrc.kaust.edu.sa/HMDARG/
