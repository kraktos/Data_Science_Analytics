import pandas as pd
from scipy.stats import mode
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# fit an algorithm and cross validate
def algo_fit_cross_validated(training_matrix, target):
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=6)

    scores_rf = cross_validation.cross_val_score(rf, training_matrix, target, cv=5)
    print("(Random Forest) Accuracy: %0.4f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))

    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                             algorithm="SAMME.R",
                             n_estimators=200)
    scores_ab = cross_validation.cross_val_score(bdt, training_matrix, target, cv=5)
    print("(ADA Boost) Accuracy: %0.4f (+/- %0.2f)" % (scores_ab.mean(), scores_ab.std() * 2))

    # SVM
    svm = SVC(kernel="linear", C=0.05)
    svm.fit(training_matrix, target)

    scores_svm = cross_validation.cross_val_score(svm, training_matrix, target, cv=5)
    print("(svm) Accuracy: %0.4f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))

    if scores_ab.mean() > scores_rf.mean():
        algo = bdt
    else:
        algo = rf

    algo.fit(training_matrix, target)
    return algo


# encode labels to numeric values
def encode(df_train, df_test):
    le_sex = preprocessing.LabelEncoder()
    feature_matrix = df_train.iloc[:, 0:12]
    target_matrix = df_train.iloc[:, 12]
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

    return df_train, df_test


# run the fitted algo on the test
def predict_save_file(df_test, test_encoded, algo):
    final_preds = pd.DataFrame({'Loan_ID': df_test.iloc[:, 0],
                                'Loan_Status': pd.Series(algo.predict(test_encoded))},
                               columns=['Loan_ID', 'Loan_Status'])

    final_preds.to_csv('data.csv', header=True, index=False)


def main():
    # clean up the inputs
    df_train, df_test = preprocess('train.csv', 'test.csv')

    # encode categorical values
    encoded_train_matrix, target, test_encoded = encode(df_train, df_test)

    # fit algorithm
    rf = algo_fit_cross_validated(encoded_train_matrix, target)

    # run on test file to predict and report submission file
    predict_save_file(df_test, test_encoded, rf)


if __name__ == "__main__":
    main()
