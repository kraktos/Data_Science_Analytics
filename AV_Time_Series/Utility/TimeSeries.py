from datetime import datetime

import pandas as pd
# parse input, to create a dataframe.
from matplotlib import pyplot as plt


def get_the_data(raw_data, given_date_format, delimiter):
    df = pd.read_csv(raw_data, delimiter=delimiter)
    df['ts'] = df['Datetime'].apply(lambda x: datetime.strptime(x, given_date_format))
    df['day'] = df['ts'].apply(lambda x: x.day)
    df['month'] = df['ts'].apply(lambda x: x.month)
    df['year'] = df['ts'].apply(lambda x: x.year)
    df = df.drop('Datetime', axis=1)

    df_by_day = df.groupby(['day', 'month', 'year']).mean()
    # sort and re-index
    df_by_day = df_by_day.reset_index(drop=False)
    df_by_day = df_by_day.sort_values(by=['year', 'month', 'day'], ascending=[1, 1, 1])
    df_by_day = df_by_day.set_index(['day', 'month', 'year'])

    df_by_month = df.groupby(['month', 'year']).mean()
    df_by_month = df_by_month.drop('day', axis=1)

    # sort and re-index
    df_by_month = df_by_month.reset_index(drop=False)
    df_by_month = df_by_month.sort_values(by=['year', 'month'], ascending=[1, 1])
    df_by_month = df_by_month.set_index(['month', 'year'])

    df_by_year = df.groupby(['year']).mean()
    df_by_year = df_by_year.drop(['day', 'month'], axis=1)

    # sort and re-index
    df_by_year = df_by_year.reset_index(drop=False)
    df_by_year = df_by_year.sort_values(by=['year'], ascending=[1])
    df_by_year = df_by_year.set_index(['year'])

    return df.set_index(['ts']), df_by_day, df_by_month, df_by_year


def plotter(data, file_name):
    ax = data.plot()
    fig = ax.get_figure()
    fig.autofmt_xdate(rotation=45)
    fig.savefig(file_name)


# output the ts
def plot_data(data, df_by_day, df_by_month, df_by_year):
    plt.figure()

    plotter(data[['Count']], 'HOURLY_output.png')
    plotter(df_by_day[['Count']], 'DAY_output.png')
    plotter(df_by_month[['Count']], 'MONTH_output.png')
    plotter(df_by_year[['Count']], 'YEAR_output.png')


def add_additional_columns(data):
    print len(data)
