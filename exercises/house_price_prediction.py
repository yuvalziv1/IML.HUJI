from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()

    # valid value
    df = df[df['view'].isin(range(5))]
    df = df[df['condition'].isin(range(1, 6))]
    df = df[df['grade'].isin(range(1, 14))]

    for col_name in ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_living15', 'sqft_lot15']:
        df = df[df[col_name] > 0]
    for col_name in ['sqft_above', 'sqft_basement']:
        df = df[df[col_name] >= 0]

    df = df[df['yr_built'] >= 1000]
    df = df[df['bedrooms'] <= 15]

    # categorical
    df['yr_built20'] = (df['yr_built'] // 20).astype(int)
    yr_built20 = pd.get_dummies(df.yr_built20)
    yr_built20.rename(columns={95: 'yr_1900', 96: 'yr_1920', 97: 'yr_1940', 98: 'yr_1960', 99: 'yr_1980', 100: 'yr_2000'}, inplace= True)
    df = df.join(yr_built20)

    df['yr_renovated'] = np.where(df['yr_renovated'] < 2000, 0, 1)

    zip_dummies = pd.get_dummies(df.zipcode, prefix='zip:')
    df = df.join(zip_dummies)

    # unnecessary features
    for col_name in ['id', 'date', 'lat', 'long', 'yr_built', 'zipcode']:
        df = df.drop(col_name, axis=1)

    y = df.price
    X = df.drop(['price'], axis=1)

    return X, y


def __pearson_correlation(x, y):
    """
    Parameters
    ----------
    x, y: vectors

    Returns
    -------
    the Pearson Correlation of x and y
    """
    return (np.cov(x, y)[0][1]) / (np.std(x) * np.std(y))


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    irrelevant_features = ['zip', 'yr_19', 'yr_20']
    for feature in X:
        if any(sub_str in str(feature) for sub_str in irrelevant_features): continue
        feature_data = X[feature]
        corr = __pearson_correlation(feature_data, y)
        fig = go.Figure([go.Scatter(x=feature_data, y=y, mode='markers')],
                        layout=go.Layout(title=f"Pearson Correlation between {feature} and price: {corr}",
                                         xaxis={"title": feature}, yaxis={"title": "price"}))
        fig.write_image(f"{output_path}/{str(feature)}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("/Users/yuval/PycharmProjects/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()

    lin_reg = LinearRegression()
    data_size = len(train_y)  # training data size
    average_loss, variance_loss, std_loss, sample_size = [], [], [], []

    for p in range(10, 101):
        p_loss = []
        p_size = int(np.ceil(p*data_size/100))
        sample_size.append(p)
        for i in range(10):
            p_train_X = train_X.sample(p_size)
            p_train_y = train_y.loc[p_train_X.index]
            p_train_X, p_train_y = p_train_X.to_numpy(), p_train_y.to_numpy()
            lin_reg.fit(p_train_X, p_train_y)
            p_loss.append(lin_reg.loss(test_X, test_y))
        average_loss.append(np.mean(p_loss))
        variance_loss.append(np.var(p_loss))  # not sure why we need it
        std_loss.append(np.std(p_loss))

    # change to numpy arrays
    sample_size = np.array(sample_size)
    average_loss = np.array(average_loss)
    variance_loss = np.array(variance_loss)
    std_loss = np.array(std_loss)

    # plot
    go.Figure([go.Scatter(x=sample_size, y=average_loss, mode="markers+lines", marker=dict(color="black", size=1), showlegend=False),
               go.Scatter(x=sample_size, y=average_loss-2*std_loss, fill=None, mode="lines", line=dict(color="lightgrey"), showlegend=False),
               go.Scatter(x=sample_size, y=average_loss+2*std_loss, fill="tonexty", mode="lines", line=dict(color="lightgrey"), showlegend=False)],
              layout=go.Layout(title=r"$\text{Average Loss as function of training size percent with error ribbon of size}$", height=400)).show()









