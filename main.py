import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, accuracy_score, precision_score
    , recall_score, roc_curve, confusion_matrix, auc, RocCurveDisplay, ConfusionMatrixDisplay
    , average_precision_score, precision_recall_curve)
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 10)

def simple_over_sample(df, column, value, copies = 2):
    targets_df = df.loc[df[column] == value]
    og_targets_df = targets_df
    for i in range(copies-1):
        targets_df = pd.concat([targets_df, og_targets_df])

    oversample_df = pd.concat([df, targets_df])
    return oversample_df

def random_over_sample(df, column, value, num_samples = 50):
    targets_df = df.loc[df['Target'] == 'Failure']
    targets_df.sample(n = num_samples, replace = True)
    oversample_df = pd.concat([df, targets_df])
    return oversample_df

def fit_classification_models(X_train, y_train):
    models = {}

    rf_params = {'n_estimators': [10, 30, 100, 300], 'max_depth': [None, 10, 30]}
    for n_est in rf_params['n_estimators']:
        for max_depth in rf_params['max_depth']:
            models['randomForest' + str(n_est) + str(max_depth)] = (
                RandomForestClassifier(n_estimators=n_est, max_depth=max_depth)
            )

    print(models)


    dt_params = {'max_depth': [None, 10, 30], 'min_samples_split': [2, 5, 10, 30]}
    for min_samp in dt_params['min_samples_split']:
        for max_depth in dt_params['max_depth']:
            models['decisionTree' + str(min_samp) + str(max_depth)] = (
                DecisionTreeClassifier(min_samples_split=min_samp, max_depth=max_depth)
            )

    print(models)

    # svc_params = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'sigmoid', 'poly']}
    # for Cval in svc_params['C']:
    #     for kernel in svc_params['kernel']:
    #         models['supportVector' + str(Cval) + str(kernel)] = (
    #             SVC(C=Cval, kernel=kernel)
    #         )
    # print(models)

    for model in models:
        print('Starting Training on ' + model)
        models[model].fit(X_train, y_train)

    return models


def predict_on_models(X_train, X_test, model_dict):
    y_hats_train = {}
    y_hats_test = {}

    for model in model_dict:
        y_hats_train[model] = model_dict[model].predict(X_train)
        y_hats_test[model] = model_dict[model].predict(X_test)

    return y_hats_train, y_hats_test

def get_probabilities(X_train, X_test, model_dict):
    class_probs_train = {}
    class_probs_test = {}

    for model in model_dict:
        class_probs_train[model] = model_dict[model].predict_proba(X_train)[:, 1]
        class_probs_test[model] = model_dict[model].predict_proba(X_test)[:, 1]

    return class_probs_train, class_probs_test


def evaluate_model_predicitions(y_train, y_test, yhats_train, yhats_test):
    prec_scores_train = {}
    recall_scores_train = {}
    f1_scores_train = {}
    accuracy_train = {}
    confusion_matrices_train = {}

    prec_scores_test = {}
    recall_scores_test = {}
    f1_scores_test = {}
    accuracy_test = {}
    confusion_matrices_test = {}

    for pred in yhats_train:
        # get training metrics
        prec_scores_train[pred] = precision_score(yhats_train[pred], y_train)
        recall_scores_train[pred] = recall_score(yhats_train[pred], y_train)
        f1_scores_train[pred] = f1_score(yhats_train[pred], y_train)
        accuracy_train[pred] = accuracy_score(yhats_train[pred], y_train)
        confusion_matrices_train[pred] = confusion_matrix(yhats_train[pred], y_train)

        #get testing metrics
        prec_scores_test[pred] = precision_score(yhats_test[pred], y_test)
        recall_scores_test[pred] = recall_score(yhats_test[pred], y_test)
        f1_scores_test[pred] = f1_score(yhats_test[pred], y_test)
        accuracy_test[pred] = accuracy_score(yhats_test[pred], y_test)
        confusion_matrices_test[pred] = confusion_matrix(yhats_test[pred], y_test)

    return (prec_scores_train, recall_scores_train, f1_scores_train, accuracy_train, confusion_matrices_train
            , prec_scores_test, recall_scores_test, f1_scores_test, accuracy_test, confusion_matrices_test)


def evaluate_model_probas():
    return None
def plot_results(model_scores, title, x_lab, y_lab, commit = False):
    plt.bar(list(model_scores.keys()), [score[0] for score in model_scores.values()], width = 0.4, align = 'edge', color = 'red', label = "Train")
    plt.bar(list(model_scores.keys()), [score[1] for score in model_scores.values()], width =-0.4, align = 'edge', color = 'blue', label = "Test")
    plt.title(title)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.xticks(rotation=90)
    plt.ylim(0,1.01)
    plt.legend()

    if commit == True:
        plt.savefig(title + ".png")
        print("Image saved to file: " + title + ".png")

    plt.show()
    return None

def main():
    # import and clean the dataset

    filepath = "predictive_maintenance.csv"
    main_df = pd.read_csv(filepath)
    main_df = main_df.set_index("Product ID", drop=True)
    main_df = main_df.drop(columns=['UDI', 'Failure Type'])

    # get dummies from Type
    main_df = pd.get_dummies(main_df)

    # oversample Failure conditions
    main_df = simple_over_sample(main_df, "Target", 'Failure', 10)

    # Parse Target Variables and Feature Set and split into test and train sets
    y = main_df["Target"]
    X = main_df.drop(columns = ["Target"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    models = fit_classification_models(X_train, y_train)
    yhats_train, yhats_test = predict_on_models(X_train, X_test, models)
    #proba_train, proba_test = get_probabilities(X_train, X_test, models)
    evals = evaluate_model_predicitions(y_train, y_test, yhats_train, yhats_test)


    f1_score_tuple = {item:(evals[2][item], evals[7][item]) for item in evals[2]}
    prec_tuple = {item:(evals[0][item], evals[5][item]) for item in evals[0]}
    recall_tuple = {item:(evals[1][item], evals[6][item]) for item in evals[1]}
    accuracy_tuple = {item:(evals[3][item], evals[8][item]) for item in evals[3]}
    # confusion_tuple = {item: (evals[4][item], evals[9][item]) for item in evals[4]}
    # for item in evals[4]:
    #     fig, ax = plt.subplots(1,2)
    #     ax[0] = ConfusionMatrixDisplay(evals[4][item]).plot()
    #     ax[1] = ConfusionMatrixDisplay(evals[9][item]).plot()
    #
    # plt.show()


    print(f1_score_tuple)
    plot_results(f1_score_tuple, "f1 scores train vs. test", "Models", "f1 score")
    plot_results(prec_tuple, "precision scores train vs. test", "Models", "precision score")
    plot_results(recall_tuple, "recall scores train vs. test", "Models", "recall score")
    plot_results(accuracy_tuple, "accuracy train vs. test", "Models", "Accuracy")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
