import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    df = df[df['Temp'] >= -30]
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df.drop(columns='Date')
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temp = load_data("/Users/yuval/PycharmProjects/IML.HUJI/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_temp = city_temp[city_temp["Country"] == "Israel"]
    israel_temp = israel_temp.drop(columns='Country')

    # plot 1
    px.scatter(israel_temp, x=israel_temp.DayOfYear, y=israel_temp.Temp, color=israel_temp.Year,
               title="Average daily temperature as function of the day of year").show()
    # plot 2
    month_std_temp = israel_temp.groupby(["Month"]).agg(np.std).Temp
    px.bar(month_std_temp, x=month_std_temp.index, y=month_std_temp, labels={'y': "temp standard deviation "},
           title="standard deviation of the daily temperatures grouped by month").show()

    # Question 3 - Exploring differences between countries
    city_grouped = city_temp.groupby(["Country", "Month"]).Temp.agg([np.std, np.mean]).reset_index()
    px.line(city_grouped, x='Month', y='mean', color='Country', error_y='std',
            title="average monthly temperature with error bars of standard deviation").show()

    # Question 4 - Fitting model for different values of `k`
    X, y = israel_temp.DayOfYear, israel_temp.Temp
    train_X, train_y, teat_X, test_y = split_train_test(X, y)
    # change to numpy array
    train_X, train_y = train_X.to_numpy(), train_y.to_numpy()
    teat_X, test_y = teat_X.to_numpy(), test_y.to_numpy()

    k_loss = []
    for k in range(1,11):
        p = PolynomialFitting(k)
        p.fit(train_X, train_y)
        p_loss = np.around(p.loss(teat_X, test_y), decimals=2)
        print(f"for k:{k}  loss = {p_loss}")
        k_loss.append(p_loss)
    # plot
    px.bar(x=range(1, 11), y=k_loss, labels={'x': "Polynom degree", 'y': "Loss"},
           title="test error recorded for each degree").show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(k=5)
    model.fit(X, y)
    other_countries_temp = city_temp[city_temp.Country != 'Israel']
    countries_array = other_countries_temp.Country.unique()

    country_loss = []
    for country in countries_array:
        country_data = other_countries_temp[other_countries_temp.Country == country]
        country_loss.append(model.loss(country_data.DayOfYear, country_data.Temp))

    # plot
    px.bar(x=countries_array, y=country_loss, labels={'x': "Country", 'y': "Loss"},
           title="model’s error over each of the other countries").show()






