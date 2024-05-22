Predictive Maintenance Project for Classification Methods on Imbalanced Data

Dataset Link: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data

### Current Results:

Best Models to classify failure without overfitting are Decision Tree classifiers with Random Forest classification performing similarly across hyperparamters although overfitting.

Evaluation is based on f1_score or the geometric mean of precision and recall for the dataset and models with test f1_score ~0.8-0.85 and training dataset

f1_score on test data is floating just below .7 for best performing ensemble of models after applying oversampling methods.

Planning to incorporate ensembling/voting methods for best performing models at various hyperparams
