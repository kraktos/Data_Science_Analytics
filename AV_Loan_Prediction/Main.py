import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from scipy.stats import mode
import xgboost as xgb

def main():
    le_sex = preprocessing.LabelEncoder()

    df_train = pd.read_csv('train.csv')
    # print(df_train.shape)

    # missing values imputing
    df_train["LoanAmount"].fillna(df_train["LoanAmount"].mean(), inplace=True)
    df_train["Loan_Amount_Term"].fillna(df_train["Loan_Amount_Term"].mean(), inplace=True)
    df_train["Credit_History"].fillna(df_train["Credit_History"].mean(), inplace=True)
    zz = mode(df_train['Gender'])
    # print(mode(df_train['Gender'])[0][0])

    df_train['Gender'].fillna(mode(df_train['Gender'])[0][0], inplace=True)
    df_train['Married'].fillna(mode(df_train['Married'])[0][0], inplace=True)
    df_train['Self_Employed'].fillna(mode(df_train['Self_Employed'])[0][0], inplace=True)

    # df_train = df_train.fillna('-')
    # print(df_train.shape)

    df_test = pd.read_csv('test.csv')
    # print(len(df_test))
    df_test["LoanAmount"].fillna(df_test["LoanAmount"].mean(), inplace=True)
    df_test["Loan_Amount_Term"].fillna(df_test["Loan_Amount_Term"].mean(), inplace=True)
    df_test["Credit_History"].fillna(df_test["Credit_History"].mean(), inplace=True)

    df_test['Gender'].fillna(mode(df_test['Gender'])[0][0], inplace=True)
    df_test['Married'].fillna(mode(df_test['Married'])[0][0], inplace=True)
    df_test['Self_Employed'].fillna(mode(df_test['Self_Employed'])[0][0], inplace=True)

    # df_test = df_test.fillna('-')

    original_id = df_test.iloc[:, 0]
    test_encoded = df_test.apply(le_sex.fit_transform)
    # print(len(df_test))

    X = df_train.iloc[:, 0:12]
    z = df_train.iloc[:, 12]
    # y = df_train.iloc[:, 12].reshape(len(df_train), 1)
    # y = y.apply(lambda x: x[0])
    # print(X.shape)
    X_encoded = X.apply(le_sex.fit_transform)
    #
    # for est in range(100, 2000, 100):
    #     for dep in range(3, 10):
    rf = RandomForestClassifier(n_estimators=200, max_depth=6)
    # gbm = xgb.XGBClassifier(max_depth=7, n_estimators=100, learning_rate=0.05).fit(X_encoded, z)

    # rf.fit(X_train_encoded, y_train)

    scores = cross_validation.cross_val_score(rf, X_encoded, z, cv = 5)
    # print("ESt = {}, Dep = {}, Score = {}".format(est, dep, scores.mean()))
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # y_pred = rf.predict(X_test_encoded)
    # score = accuracy_score(y_test, y_pred)

    # print(score)
    rf.fit(X_encoded, z)
    Loan_ID = test_encoded.iloc[:, 0]
    Loan_Status = rf.predict(test_encoded)

    # print(len(Loan_ID))
    # print(len(Loan_Status))
    # print(df_test.shape)

    final_preds = pd.DataFrame({'Loan_ID': original_id,
                                    'Loan_Status': pd.Series(Loan_Status)}, columns=['Loan_ID', 'Loan_Status'])

    final_preds.to_csv('data.csv', header=True, index=False)


if __name__ == "__main__":
    main()
