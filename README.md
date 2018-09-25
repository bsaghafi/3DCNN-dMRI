# 3DCNN-dMRI
Keras Implementation of a 3DCNN that predicts level of head impact exposure from changes in FA images post-pre.
This repository includes training, validation, permutation testing and occlusion map for discovering feature importance.

Please cite the following paper if you use the code:
* [Quantifying the Association between White Matter Integrity Changes and Subconcussive Head Impact Exposure ](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10575/105750E/Quantifying-the-association-between-white-matter-integrity-changes-and-subconcussive/10.1117/12.2293023.short)

## Training and Testing
For training and testing the model, run:
```
3DCNN_diffusion_train_test.py
```
Total run time: ~42 minutes on Tesla K80.


3DCNN_diffusion_test_pretrained.py 
Purpose: Tests a pretrained 3DCNN model to predict the level of head impact exposure from changes in FA images post-pre. The model (previously trained on 48 subjects) is tested on 12 subjects. 
Total run time: ~14 seconds on Tesla K80.


3DCNN_diffusion_ModelSelection_validation.py 
Purpose: Validate a 3DCNN model for model selection using 5-fold cross validation. In each fold, It trains on 38 subjects and validates on 10. An adaptive learning rate was used. In each validation test, the learning rate was reduced by 50% if the ROC AUC did not improve for 10 epochs. An early stopping schedule with look ahead was developed. Training stops when the network shows no improvement in AUC score for 50 epochs. 
Outputs: Average classification performance measures (AUC and F1-score) on the validation data as well as average best epochs over the 5 folds. 
Total run time:  ~24 minutes on Tesla K80. 


permutation_testing.py 
Purpose: performs permutation testing to evaluate the statistical significance of the overall model 
Inputs: delta FA maps (Post-Pre) and their classification labels, the 3DCNN model, the accuracy of the model on the test data 
Parameters: Number of permutation tests (n_permutations=100) 
Outputs: permutation scores, p-value, Probability density function (histogram) 
Total run time: ~3 days for 100 permutation tests on Tesla K80
 

occlusion_map.py 
Purpose: computes occlusion maps for the high impact exposure sample 
Inputs: delta FA maps (Post-Pre) and their classification labels, the 3DCNN model, the subject to compute the occlusion map for 
Outputs: occlusion map which can be overlaid on the FA map 
Total run time: ~3 minutes on Tesla K80.
