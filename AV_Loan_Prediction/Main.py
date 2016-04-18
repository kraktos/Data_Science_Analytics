import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC


def func(row):
    if row['Dependents']:
        return row['total_income'] / row['Dependents']
    else:
        return 0


# fit an algorithm and cross validate
def algo_fit_cross_validated(training_matrix, target):
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(training_matrix, target)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    l = list(training_matrix.columns.values)
    for f in range(training_matrix.shape[1]):
        print("%d. feature %d(%s) (%f)" % (f + 1, indices[f], l[indices[f]], importances[indices[f]]))

    ##### Works well ######
    # SVM
    # svm = SVC(kernel="linear", C=0.06)
    # svm.fit(training_matrix, target)
    #
    # scores_svm = cross_validation.cross_val_score(svm, training_matrix, target, cv=5)
    # print("(svm) Accuracy: %0.5f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))
    #
    # return svm
    ##### Works well ######

    # Random Forest
    rf = RandomForestClassifier(n_estimators=1500, max_depth=2, max_features=4)
    scores_rf = cross_validation.cross_val_score(rf, training_matrix, target, cv=5)
    print("(Random Forest) Accuracy: %0.5f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))
    rf.fit(training_matrix, target)
    return rf

    # Create and fit an AdaBoosted decision tree
    # bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
    #                          algorithm="SAMME.R",
    #                          n_estimators=600)
    # scores_ab = cross_validation.cross_val_score(bdt, training_matrix, target, cv=5)
    # print("(ADA Boost) Accuracy: %0.4f (+/- %0.2f)" % (scores_ab.mean(), scores_ab.std() * 2))
    # bdt.fit(training_matrix, target)
    #
    # return bdt

    # Decision trees
    # dt = tree.DecisionTreeClassifier(max_features=6, max_depth=4)
    # scores_rf = cross_validation.cross_val_score(dt, training_matrix, target, cv=5)
    # print("(Decision Trees) Accuracy: %0.4f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))
    #
    # dt.fit(training_matrix, target)
    # return dt

    # XGBoost
    # gbm = xgb.XGBClassifier(max_depth=4, n_estimators=200, learning_rate=0.05).fit(training_matrix, target)
    # scores_xgb = cross_validation.cross_val_score(gbm, training_matrix, target, cv=5)
    # print("(XGBoost) Accuracy: %0.4f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std() * 2))
    #
    # return gbm


# encode labels to numeric values
def encode(df_train, df_test):
    cnt = len(df_train.columns) - 1

    le_sex = preprocessing.LabelEncoder()
    feature_matrix = df_train.iloc[:, 0:cnt]
    target_matrix = df_train.iloc[:, cnt]
    encoded_train_feature_matrix = feature_matrix.apply(le_sex.fit_transform)
    encoded_test_feature_matrix = df_test.apply(le_sex.fit_transform)
    return encoded_train_feature_matrix, target_matrix, encoded_test_feature_matrix


# read the training file and test files to create preprocessed dataframes
def preprocess(train_file, test_file):
    df_train = pd.read_csv(train_file)

    # missing values imputing
    df_train["LoanAmount"].fillna(df_train["LoanAmount"].mean(), inplace=True)
    df_train["Loan_Amount_Term"].fillna(df_train["Loan_Amount_Term"].mean(), inplace=True)
    df_train["Credit_History"].fillna(df_train["Credit_History"].mean(), inplace=True)

    df_train['Gender'].fillna(mode(df_train['Gender'])[0][0], inplace=True)
    df_train['Married'].fillna(mode(df_train['Married'])[0][0], inplace=True)
    df_train['Self_Employed'].fillna(mode(df_train['Self_Employed'])[0][0], inplace=True)

    df_test = pd.read_csv(test_file)
    df_test["LoanAmount"].fillna(df_test["LoanAmount"].mean(), inplace=True)
    df_test["Loan_Amount_Term"].fillna(df_test["Loan_Amount_Term"].mean(), inplace=True)
    df_test["Credit_History"].fillna(df_test["Credit_History"].mean(), inplace=True)

    df_test['Gender'].fillna(mode(df_test['Gender'])[0][0], inplace=True)
    df_test['Married'].fillna(mode(df_test['Married'])[0][0], inplace=True)
    df_test['Self_Employed'].fillna(mode(df_test['Self_Employed'])[0][0], inplace=True)

    tmp_train = df_train['Loan_Status']

    # add features
    df_train['total_income'] = df_train['ApplicantIncome'] + df_train['CoapplicantIncome']
    df_test['total_income'] = df_test['ApplicantIncome'] + df_test['CoapplicantIncome']

    df_train['ratio'] = df_train['total_income'] / df_train['LoanAmount']
    df_test['ratio'] = df_test['total_income'] / df_test['LoanAmount']

    df_train['emi'] = df_train['LoanAmount'] / df_train['Loan_Amount_Term']
    df_test['emi'] = df_test['LoanAmount'] / df_test['Loan_Amount_Term']

    # drop redundant features
    ids = df_test['Loan_ID']

    df_train.drop('Loan_Status', axis=1, inplace=True)

    df_train.drop('Loan_ID', axis=1, inplace=True)
    df_test.drop('Loan_ID', axis=1, inplace=True)

    df_train.drop('ApplicantIncome', axis=1, inplace=True)
    df_test.drop('ApplicantIncome', axis=1, inplace=True)

    df_train.drop('CoapplicantIncome', axis=1, inplace=True)
    df_test.drop('CoapplicantIncome', axis=1, inplace=True)

    df_train.drop('Loan_Amount_Term', axis=1, inplace=True)
    df_test.drop('Loan_Amount_Term', axis=1, inplace=True)

    df_train.drop('LoanAmount', axis=1, inplace=True)
    df_test.drop('LoanAmount', axis=1, inplace=True)

    df_train.drop('Education', axis=1, inplace=True)
    df_test.drop('Education', axis=1, inplace=True)

    df_train.drop('Married', axis=1, inplace=True)
    df_test.drop('Married', axis=1, inplace=True)

    df_train.drop('Gender', axis=1, inplace=True)
    df_test.drop('Gender', axis=1, inplace=True)

    df_train.drop('Self_Employed', axis=1, inplace=True)
    df_test.drop('Self_Employed', axis=1, inplace=True)

    df_train.drop('Property_Area', axis=1, inplace=True)
    df_test.drop('Property_Area', axis=1, inplace=True)

    # df_train.drop('Dependents', axis=1, inplace=True)
    # df_test.drop('Dependents', axis=1, inplace=True)

    df_train.insert(len(df_train.columns), 'Loan_Status', value=tmp_train)

    return ids, df_train, df_test


# run the fitted algo on the test
def predict_save_file(ids, test_encoded, algo):
    final_preds = pd.DataFrame({'Loan_ID': ids,
                                'Loan_Status': pd.Series(algo.predict(test_encoded))},
                               columns=['Loan_ID', 'Loan_Status'])

    final_preds.to_csv('data.csv', header=True, index=False)


def main():
    # clean up the inputs
    ids, df_train, df_test = preprocess('train.csv', 'test.csv')

    # encode categorical values
    encoded_train_matrix, target, test_encoded = encode(df_train, df_test)

    # fit algorithm
    rf = algo_fit_cross_validated(encoded_train_matrix, target)

    # run on test file to predict and report submission file
    predict_save_file(ids, test_encoded, rf)


if __name__ == "__main__":
    main()
