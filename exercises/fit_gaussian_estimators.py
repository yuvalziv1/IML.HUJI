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
    pdfs = uni_gauss.pdf(X)
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{PDF of the previously drawn samples }$",
                  xaxis_title="$\\text{  sample values}$",
                  yaxis_title="$\\text{  PDF values}$",
                  height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    cov = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, size=(1000,))
    multi_gauss = MultivariateGaussian()
    multi_gauss.fit(X)
    print("estimated expectation: ", multi_gauss.mu_)
    print("covariance matrix: ", multi_gauss.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_likelihoods = []
    for i in range(200):
        log_likelihoods.append([])
        for j in range(200):
            log_likelihoods[i].append(MultivariateGaussian.log_likelihood(np.array([f1[i], 0, f3[j], 0]), cov, X))
    heatmap = go.Figure(data=go.Heatmap(x=f3, y=f1, z=log_likelihoods))
    heatmap.update_layout(showlegend=True, autosize=True, title=" heatmap ",
                          xaxis_title="$\\text{  f2 values}$",
                            yaxis_title="$\\text{  f1 values}$" )
    heatmap.show()

    # Question 6 - Maximum likelihood
    argmax = np.argmax(log_likelihoods)
    i = argmax // 200
    j = argmax % 200
    f1_max = f1[i]
    f3_max = f3[j]
    max = MultivariateGaussian.log_likelihood(np.array([f1_max, 0, f3_max, 0]), cov, X)
    print("max value: ", max)
    print("f1: ", f1_max, "f3: ", f3_max)

    # QUIZ
    quiz_q3 = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
          -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])

    print(UnivariateGaussian.log_likelihood(10, 1, quiz_q3))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
