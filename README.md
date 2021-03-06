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

For testing the pretrained model, run: 
```
3DCNN_diffusion_test_pretrained.py
```

For performing model selection using 5-fold cross validation, run:
```
3DCNN_diffusion_ModelSelection_validation.py
```
In each validation test, the learning rate was reduced by 50% if the ROC AUC did not improve for 10 epochs. An early stopping schedule with look ahead was developed. Training stoped when the network showed no improvement in AUC score for 50 epochs. 

## Permutation Testing
For performing permutation testing and evaluating the statistical significance of the overall model, run:
```
permutation_testing.py
```

## Occlusion Maps
For computing occlusion maps for the high impact exposure sample, run:
```
occlusion_map.py
```
