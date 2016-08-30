"""
perform risk modelling
"""
import pandas as pd
from sklearn import cross_validation as cv
import numpy
from sklearn import preprocessing
from sklearn import linear_model, datasets, metrics
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import math
import itertools
from sklearn import metrics


def get_data(type_of_data):
    """
    load the data in memory
    """
    if type_of_data == 'train':
        return pd.read_csv('cs-training.csv')
    else:
        return pd.read_csv('cs-test.csv')


def split_feature_target(df):
    """
    split the data into feature/target
    """

    # helper function to split into feature and target and also do hot encoding
    feature_variable = df[[i for i in list(df.columns) if i != 'SeriousDlqin2yrs']]
    target_variable = pd.DataFrame(df, columns=['SeriousDlqin2yrs'])

    return feature_variable, target_variable

    # return cv.train_test_split(feature_variable, target_variable, test_size=0.3)


def feature_engineering(df):
    # get the column names
    names = df.columns.values
    # for val in names:
    #     print "{}:SeriousDlqin2yrs = {}".format(val,
    #                                             math.fabs(numpy.corrcoef(df[val], df['SeriousDlqin2yrs'])[0][1]))

    # del df['NumberRealEstateLoansOrLines']
    # del df['NumberOfOpenCreditLinesAndLoans']
    del df['RevolvingUtilizationOfUnsecuredLines']
    # del df['NumberOfDependents']

    return df


def pre_processing(df):
    """
    preprocess the given dataframe to remove redundant features
    """
    # check for null values
    df2 = df[numpy.isfinite(df['MonthlyIncome'])]

    # percent null values
    drop = 1 - (len(df2)/float(len(df)))

    # if more than 5% of data lost, impute
    if drop > 0.05:
        df = df.fillna(df.mean())
    else:
        df = df2

    # standardize the data first
    #df_scaled = preprocessing.scale(df)

    # print the list of features and their variances
    # names = df.columns.values
    # for val in names:
    #     print val, " ==> ", numpy.var(df_scaled[val])

    # del df['NumberOfDependents']

    return df


def up_sample(user_df, threshold):
    """
    up-sample a given data set till the positive class labels have threshold percentage of representation
    """
    print ("Up sampling data at {}".format(threshold))

    # under represented class
    under_class_df = user_df.ix[(user_df['SeriousDlqin2yrs'] == 1)]

    new_balance = 0

    # while the ratio is no balanced
    while new_balance < threshold:
        sampled_under_class_df = under_class_df.sample(n=min(500, len(under_class_df)))

        # add random data rows for the under represented class
        user_df = pd.concat([user_df, sampled_under_class_df])

        # calculate the new ratio of the class labels
        new_balance = sum(user_df['SeriousDlqin2yrs'].values) / float(len(user_df))

    return user_df


def gen_model(X, y):

    # logistic regression ##############################################################
    # logistic = linear_model.LogisticRegression()
    # model = logistic.fit(X_train, y_train)
    # print model.score(X_test, y_test)
    #
    # # Confusion matrix table ###########################################################
    # y_actu = pd.Series(y_test['SeriousDlqin2yrs'].values, name='Actual')
    # y_pred = pd.Series(model.predict(X_test), name='Predicted')
    # df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    # print df_confusion
    #
    # # F-1 score ########################################################
    # print("F1 = {}".format(f1_score(y_actu, y_pred, average='binary')))

    ####################################################################################
    # random forest regression ##############################################################
    rf = RandomForestClassifier(n_estimators=80, n_jobs=4)
    scores = cv.cross_val_score(rf, X, numpy.ravel(y)
                                , cv = 5, scoring = 'f1')

    # model = rf.fit(X_train, numpy.ravel(y_train))
    # acc =  model.score(X_test, y_test)

    # Confusion matrix table ###########################################################
    # y_actu = pd.Series(y_test['SeriousDlqin2yrs'].values, name='Actual')
    # y_pred = pd.Series(model.predict(X_test), name='Predicted')
    # df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    # print df_confusion

    # F-1 score ########################################################
    # print("F1 = {}".format(f1_score(y_actu, y_pred, average='binary')))
    # f1 = f1_score(y_actu, y_pred, average='binary')

    return scores.mean()


def run_all(threshold):
    """
    run the risk modelling, training, testing and crating the submissions file
    """
    # grab the raw datasets
    training_data = get_data("train")
    test_data_holdout = get_data("test")

    del training_data['Unnamed: 0']
    del test_data_holdout['Unnamed: 0']

    training_data = pre_processing(training_data)

    # feature engineering
    training_data = feature_engineering(training_data)

    # iterate all the possible feature space
    lst = training_data.columns.values
    lst = numpy.delete(lst, 0)

    temp = 0

    # get ratio of unbalanced class labels
    labels_ratio = sum(training_data['SeriousDlqin2yrs'].values) / float(len(training_data))

    # if class labels are skewed
    if labels_ratio < threshold:
        training_data = up_sample(training_data, threshold)

    for i in xrange(6, len(lst) + 1):
        els = [list(x) for x in itertools.combinations(lst, i)]
        for l in els:
            l.append('SeriousDlqin2yrs')
            x = training_data.loc[:, training_data.columns.isin(l)]

            # # get ratio of unbalanced class labels
            # labels_ratio = sum(training_data['SeriousDlqin2yrs'].values) / float(len(training_data))
            #
            # # if class labels are skewed
            # if labels_ratio < threshold:
            #     x = up_sample(x, threshold)

            # split features and target variables
            # X_train, X_test, y_train, y_test = split_feature_target(x)
            X, y = split_feature_target(x)

            # take the train test data sets to validate the model
            f1 = gen_model(X, y)
            # print l

            if f1 > temp:
                temp = f1
                print l, temp

if __name__ == '__main__':
    run_all(0.31)
