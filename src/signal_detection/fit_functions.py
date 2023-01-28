'''Functions for testing & comparing model fits.'''
import numpy as np
from scipy import stats
from typing import Union

array_like = Union[list, tuple, np.ndarray]

def log_likelihood(f_obs: Union[float, array_like], p_exp: Union[float, array_like]) -> float:
    """Computes the Log Likelihood 

    This is the log likelihood function that appears on page 145 of Dunn (2011) 
    and is also used in the ROC toolbox of Koen, Barrett, Harlow, & Yonelinas 
    (2017; see https://github.com/jdkoen/roc_toolbox). The calculation is:

        $\sum_i^{j}O_i\log(P_i)$ where $j$ refers the number of response 
        categories.

    Parameters
    ----------
    f_obs : array_like
        The observed frequencies (counts; non-cumulative) for each of the 
        response categories.
    p_exp : array_like
        The expected probabilities (non-cumulative) for each of the response 
        categories.

    Returns
    -------
    log_likelihood : float
        The log likelihood value for the given inputs.

    """

    return (np.array(f_obs) * np.log(np.array(p_exp))).sum()


def squared_errors(observed: array_like, expected: array_like) -> float:
    """Computes the sum of squared errors between observed values and those 
    which were computed by the model.

    Parameters
    ----------
    observed : array_like
        Array of observed values.
    expected : array_like
        Array of expected (model-predicted) values

    Returns
    -------
    np.array
        An array, equal to the length of the inputs, containing the computed 
        squared error values.
    """
    return (observed - expected)**2


def aic(k: int, LL: float) -> float:
    """Computes Akaike's information criterion (AIC; https://en.wikipedia.org/wiki/Akaike_information_criterion).

    Is an estimator of quality of each model relative to others, enabling model 
    comparison and selection.

    Parameters
    ----------
    k : int
        The number of estimated parameters in the model.
    LL : float
        The log-likelihood value (see `log_likelihood`).

    Returns
    -------
    float
        The AIC score.
    """
    return 2 * k - 2 * LL


def bic(k: int, n: int, LL: float) -> float:
    """Computes the Bayesian information criterion (BIC; https://en.wikipedia.org/wiki/Bayesian_information_criterion).

    Is an estimator of quality of each model relative to others, enabling model 
    comparison and selection.

    Parameters
    ----------
    k : int
        The number of estimated parameters in the model.
    n : int
        The number of data points in the observed data.
    LL : float
        The log-likelihood value (see `log_likelihood`).

    Returns
    -------
    float
        The BIC score.

    """
    return k * np.log(n) - 2 * LL
