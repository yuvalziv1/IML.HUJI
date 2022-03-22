from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uni_gauss = UnivariateGaussian()
    X = np.random.normal(10,1,1000)
    uni_gauss.fit(X)
    print((uni_gauss.mu_, uni_gauss.var_))

    # Question 2 - Empirically showing sample mean is consistent
    ms = []
    for i in range(1, 101):
        ms.append(i*10)
    dist = []
    for m in ms:
        estimated_exp = np.mean(X[:m])
        dist.append(np.abs(estimated_exp - uni_gauss.mu_))
    go.Figure([go.Scatter(x=ms, y=dist, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Absolute distance between the estimated- and true value of the expectation}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$|\hat\mu-\mu|$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
