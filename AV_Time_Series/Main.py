from AV_Time_Series.Utility.TimeSeries import get_the_data, plot_data


def main():
    data, df_by_day, df_by_month, df_by_year = get_the_data('Train.csv', '%d-%m-%Y %H:%M', ',')

    plot_data(data, df_by_day, df_by_month, df_by_year)


if __name__ == "__main__":
    main()
