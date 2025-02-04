from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        
        self.mu_ = np.mean(X)
        if self.biased_ == True:
            self.var_ = np.var(X)
        else: self.var_ = np.var(X, ddof=1)
        self.fitted_ = True
        return self

    
    @staticmethod
    def __calc_pdf(sample, mu, var):
        """
        Calculate PDF of one sample under Gaussian model with given estimators
        
        Parameters
        ----------
        sample: float
        Sample to calculate PDF for
        mu: float
        given expectation
        var: float
        given variance
        
        Returns
        -------
        pdf: float
        Calculated value of given sample for PDF function of N(mu_, var_)
        """
        
        power = (-1/2)*((sample - mu)/np.sqrt(var))**2
        coeff = 1/(np.sqrt((var)*2*np.pi))
        pdf = coeff * np.exp(power)
        return pdf
        
    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
                 
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        pdfs = np.ndarray(shape = np.shape(X))
        i = 0
        for sample in X:
            pdfs[i] = UnivariateGaussian.__calc_pdf(sample, self.mu_, self.var_)
            i+= 1
        return pdfs
                  
                

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        val = 0
        for sample in X:
           val = val + np.log((UnivariateGaussian.__calc_pdf(sample, mu, sigma)))
        return val


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    @staticmethod
    def __calc_pdf_multi(sample, mu, cov, d):
        power = (-1 / 2) * np.matmul((np.matmul((sample - mu).T, inv(cov))), (sample - mu))
        coeff = 1 / (np.sqrt(det(cov) * (2 * np.pi)**d))
        pdf = coeff * np.exp(power)
        return pdf

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        pdfs = np.ndarray(shape=np.shape(X))
        d = X.shape[1]
        i = 0
        for sample in X:
            pdfs[i] = MultivariateGaussian.__calc_pdf_multi(sample, self.mu_, self.cov_, d)
            i += 1
        return pdfs


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        a_val = np.sum((X-mu) @ inv(cov) * (X-mu))
        m = X.shape[0]
        d = X.shape[1]
        b_val = m * np.log(((2*np.pi)**d) * det(cov))
        return (-1/2)*(a_val+b_val)



