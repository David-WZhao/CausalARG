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
In addtion, we also provide a Colab link(https://colab.research.google.com/drive/1lzLFccXoGSYej1iVb1Y3TGuGfvF-EDDi?usp=sharing) to use our pre-trained model online. Through this link, you can directly access the code we provide for calling our trained model.

First, you need to download the `mode.pth` in our github link and upload it to colab as following picture.



![image](https://github.com/David-WZhao/CausalARG/assets/31216817/a49bedb4-cd77-4ddc-9f40-e0bd5eccda19)

![image](https://github.com/David-WZhao/CausalARG/assets/31216817/09e746b2-35fa-4dcc-9555-dab12607722a)




You can also open the `triained-model.ipynb` to use our trained model.



### The time of training 

The training time on a server with an RTX 3090 24GB GPU is approximately 

# Introduction of Baselines
To comprehensively evaluate the effectiveness of the proposed method, the following three representative methods are selected as baselines for performance comparison:

## BestHit
This method is conducted by comparing the sample sequences with existing ARGs in CARD by applying the BLAST or DIAMOND, and the predicted properties are assigned to samples through applying a similarity cutoff. Note that BestHit can be used only for predicting antibiotic classes and resistance mechanisms of ARGs. For detailed usage of BestHit, please refer to the link:[The Comprehensive Antibiotic Resistance Database (mcmaster.ca)](https://card.mcmaster.ca/analyze/blast)

## DeepARG
This method is a deep learning-based model which is trained by taking the consistency distribution of homologies between sample sequences and all known ARGs as input features. Note that DeepARG can be used only for predicting antibiotic classes of ARGs. For detailed usage of the DeepARG, please refer to the link: https://github.com/gaarangoa/deeparg.

## HMD-ARG
This method extracts features from raw sequences through an end-to-end deep CNN-based framework for predicting properties of ARGs. Note that HMD-ARG is a multi-task model, which can be used for predicting all of three properties of ARGs. For detailed usage of the **HMD-ARG:**, please refer to the link: [http://www.cbrc.kaust.edu.sa/HMDARG](http://www.cbrc.kaust.edu.sa/HMDARG/).
