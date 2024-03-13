# Causal-ARG

This is the code of the Causal-ARG in our manuscript
### Requirement
- python 3.8
- torch == 2.1.0
- scikit-learn == 1.2.2
- numpy == 1.22.4
- pandas == 2.1.2

### Data
The process of collecting data is described in our manuscript and metadata can be accessed upon request.

### How to run the code?
We provide two ways to use our model. First, complete the train and test via an integrated ipynb file “run.ipynb”; Second, employ our trained model to complete ARGs annotation via a Colab link.

#### Using “run.ipynb”
1. Install required python packages.
2. Download the codes to local machine.
3. Open the ipynb file “run.ipynb” in jupyter notebook, and execute all code blocks to derive prediction results.

#### Using trained model
1. Open the Colab link we provided (https://colab.research.google.com/drive/1lzLFccXoGSYej1iVb1Y3TGuGfvF-EDDi?usp=sharing).
2. Download the trained model in our github link, i.e. file "model.pth".
3. Upload the "model.pth" to the file directory in Coalb file, as shown in the following pictures.

![image](https://github.com/David-WZhao/CausalARG/assets/31216817/a49bedb4-cd77-4ddc-9f40-e0bd5eccda19)

![image](https://github.com/David-WZhao/CausalARG/assets/31216817/09e746b2-35fa-4dcc-9555-dab12607722a)


4. After uploading the "model.pth", execute all code blocks in Colab file to derive prediction results.


### The time of training 

It will take xxx minutes for training our model on a Linux server with a GPU 3090 (24GB), a CPU Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz and 64 GB RAM.  

# Introduction of Baselines
To comprehensively evaluate the effectiveness of the proposed method, the following three representative methods are selected as baselines for performance comparison:

## BestHit
This method is conducted by comparing the sample sequences with existing ARGs in CARD by applying the BLAST or DIAMOND, and the predicted properties are assigned to samples through applying a similarity cutoff. Note that BestHit can be used only for predicting antibiotic classes and resistance mechanisms of ARGs. For detailed usage of BestHit, please refer to the link:[The Comprehensive Antibiotic Resistance Database (mcmaster.ca)](https://card.mcmaster.ca/analyze/blast)

## DeepARG
This method is a deep learning-based model which is trained by taking the consistency distribution of homologies between sample sequences and all known ARGs as input features. Note that DeepARG can be used only for predicting antibiotic classes of ARGs. For detailed usage of the DeepARG, please refer to the link: https://github.com/gaarangoa/deeparg.

## HMD-ARG
This method extracts features from raw sequences through an end-to-end deep CNN-based framework for predicting properties of ARGs. Note that HMD-ARG is a multi-task model, which can be used for predicting all of three properties of ARGs. For detailed usage of the **HMD-ARG:**, please refer to the link: [http://www.cbrc.kaust.edu.sa/HMDARG](http://www.cbrc.kaust.edu.sa/HMDARG/).
