# Knowledge Enhanced Zero-shot Visual Relationship Detection

Code for paper 'Knowledge Enhanced Zero-shot Visual Relationship Detection'.


# Introduction

The model comprises two modules: logic tensor networks encoded negative domain of semantic and spatial knowledge, and a commonsense knowledge graph module updated by local spatial structure as positive domain semantic knowledge. Predictions are further constrained by region connection calculus (RCC). 
# Using Code
-   The  `models`  folder contains the trained grounded theories of the experiments;
-   The  `Visual-Relationship-Detection-master`  folder contains the object detector model and the evaluation code provided in  [https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)  for the evaluation of the phrase, relationship and predicate detection tasks on the VRD.
-  The  `data`  folder contains the data which can be downloaded from https://cs.stanford.edu/people/ranjaykrishna/vrd/
-  The  ConceptNet can be downloaded from https://github.com/commonsense/conceptnet-numberbatch

## Requirements

  The packages needed in training can be downloaded following :

     pip install -r requirements.txt

## Training

Use the complete model:  
```sh  
$ python train_all.py
```  
Use LTNs  :
```sh  
$ python train.py
```  
Use without spatial knowledge :
```sh  
$ python train_mul.py
```  
Use without CKG module :
```sh  
$ python train_RCC.py
```  
  
- The trained models are saved in the `models` folder in the files `KB_wc_2500.ckpt` (with constraints). The number in the filename (`2500`) is a parameter in the code to set the number of iterations.
## Evaluating

To run the evaluation use the following commands  
```sh  
$ python predicate_detection_mul.py$ python relationship_phrase_detection_mul.py
```  
Then, launch Matlab, move into the `Visual-Relationship-Detection-master` folder, execute the scripts `predicate_detection_LTN.m` and `relationship_phrase_detection_LTN.m` and see the results.

## Acknowledgement
This repository is based on our references [\[3\]](https://github.com/MIRALab-USTC/KG-TACT) and [\[5\]](https://github.com/ivanDonadello/Visual-Relationship-Detection-LTN)

[3] Chen, J., He, H., Wu, F., Wang, J.: Topology-aware correlations between relations for inductive link prediction in knowledge graphs. In: AAAI. vol. 35, pp. 6271–6278 (2021)

[5] Donadello, I., Serafini, L.: Compensating supervision incompleteness with prior knowledge in semantic image interpretation. In: IJCNN. pp. 1–8. IEEE (2019).
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTExNzk1MTYwM119
-->
