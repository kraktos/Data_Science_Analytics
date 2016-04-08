import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    le_sex = preprocessing.LabelEncoder()

    df_train = pd.read_csv('train.csv')
    print(df_train.shape)
    df_train = df_train.replace(r'\s+', '-', regex=True)
    print(df_train.shape)

    df_test = pd.read_csv('test.csv')
    print(len(df_test))
    df_test = df_test.fillna('-')
    original_id = df_test.iloc[:, 0]
    df_test = df_test.fillna(0)
    test_encoded = df_test.apply(le_sex.fit_transform)
    print(len(df_test))

    X = df_train.iloc[:, 0:12]
    y = df_train.iloc[:, 12].reshape(len(df_train), 1)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # to convert into numbers

    X_train_encoded = X_train.apply(le_sex.fit_transform)
    X_test_encoded = X_test.apply(le_sex.fit_transform)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_encoded, y_train)

    y_pred = rf.predict(X_test_encoded)
    score = accuracy_score(y_test, y_pred)

    print(score)

    Loan_ID = test_encoded.iloc[:, 0]
    Loan_Status = rf.predict(test_encoded)

    print(len(Loan_ID))
    print(len(Loan_Status))
    print(df_test.shape)

    percentile_list = pd.DataFrame({'Loan_ID': original_id,
                                    'Loan_Status': pd.Series(Loan_Status)}, columns=['Loan_ID', 'Loan_Status'])

    percentile_list.to_csv('data', header=True, index=False)


if __name__ == "__main__":
    main()
