import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats
from typing import Union, Optional

numeric = Union[int, float, np.number]
array_like = Union[list, tuple, np.ndarray]

def arrays_equal_length(a: array_like, b: array_like):
    if len(a) != len(b):
        return False
    else:
        return True

def keyval_table(**kwargs):
    t = PrettyTable([0,1])
    for key, val in kwargs.items():
        if isinstance(val, np.ndarray):
            for i, x in enumerate(val):
                t.add_row([f"{key} {i+1}", x])
        else:
            t.add_row([key, val])
    return t

def accumulate(arr: array_like):
    return np.cumsum(arr)

def deaccumulate(arr: array_like) -> np.ndarray:
    return np.diff(np.insert(arr, 0, 0)) # insert a 0 at the start

def compute_proportions(
    arr: array_like,
    corrected: Optional[bool]=True,
    truncate: Optional[bool]=True
    ) -> np.ndarray:
    """Compute the proportions of a response array.
    
    The input should be response counts for each criterion category for either 
    signal OR noise datasets.
    
    Parameters
    ----------
    arr : array_like
        The input array. EITHER: all responses to signal trials, OR all 
        responses to noise trials.
    corrected : bool, optional
        If True, adds a small amount, equal to i/n (where i is the index of the 
        array and `n` is the number of elements in the array) to each 
        accumulated value, and also adds 1 to the total number of responses 
        defined as the sum of the un-accumulated array (or the final element of 
        the accumulated array). The default is True.
    truncate : bool, optional
        Whether to remove the final element of the returned array. This is 
        typically required because (1) this value is always equal to 1 and is 
        therefore implied, and (2) a value of 1 cannot be converted to a 
        z-score, which is required to convert the resulting output from ROC- to
        z-space. The default is True.

    Raises
    ------
    ValueError
        If the last element of the resulting array is not equal to 1.

    Returns
    -------
    np.ndarray
        The accumulated array of proportions.
    
    Example
    -------
    >>> s = [505, 248, 226, 172, 144, 93]
    >>> compute_proportions(s)
    array([0.3636909 , 0.54235661, 0.70518359, 0.82913367, 0.93292537])
    
    >>> compute_proportions(s, corrected=False)
    array([0.36383285, 0.5425072 , 0.70533141, 0.82925072, 0.93299712])
    
    >>> compute_proportions(s, truncate=False)
    array([0.3636909, 0.54235661, 0.70518359, 0.82913367, 0.93292537, 1])

    """
    a = accumulate(arr)
    
    if corrected:
        f = [(x + i / len(a)) / (max(a) + 1) for i, x in enumerate(a, start=1)]
    else:
        f = list(a / max(a))
    
    if f[-1] != 1:
        raise ValueError(f"Expected max accumulated to be 1 but got {f[-1]}.")
    
    if truncate:
        f.pop()

    return np.array(f)


def plot_roc(
        signal: array_like,
        noise: array_like,
        ax: Optional[Axes]=None,
        chance: Optional[bool]=True,
        **kwargs
    ) -> Axes:
    """A utility to plot ROC curves. Requires signal and noise arrays in 
    probability space. Accepts scatter plot keyword arguments.
    
    Parameters
    ----------
    signal : array_like
        Signal array in probability space.
    noise : array_like
        Noise array in probability space.
    ax : Optional[Axes], optional
        Matplotlib Axes object to plot to, if already defined. The default is 
        None.
    chance : Optional[bool], optional
        Whether or not to plot the diagonal chance line (0, 0), (1, 1). The 
        default is True.
    **kwargs : TYPE
        Keyword arguments for the matplotlib.pyplot.scatter function. See 
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html.
    
    Returns
    -------
    ax : Axes
        A matplotlib Axes object with the plotted signal & noise data in 
        probability space.

    """
    if ax is None:
        fig, ax = plt.subplots()
    
    if chance:
        ax.plot([0,1], [0,1], c='k', lw=1, ls='dashed')
    
    ax.scatter(noise, signal, **kwargs, zorder=1e10)
    ax.axis('square')
    ax.set(xlabel='FP', ylabel='TP')
    return ax

def plot_zroc(
        signal: array_like,
        noise: array_like,
        ax: Optional[Axes]=None,
        reg: Optional[bool]=True,
        poly: Optional[int]=1,
        data: Optional[bool]=True,
        **kwargs
    ):
    """A utility to plot z-ROC curves. Requires signal and noise arrays in 
    probability space. Accepts scatter plot keyword arguments.
    
    Parameters
    ----------
    signal : array_like
        Signal array in probability space.
    noise : array_like
        Noise array in probability space.
    ax : Optional[Axes], optional
        Matplotlib Axes object to plot to, if already defined. The default is 
        None.
    reg : Optional[bool], optional
        Whether or not to draw a regression line. If True, see `poly`. The 
        default is True.
    poly : Optional[int], optional
        The order of the polynomial regression line. The 
        default is 1 (linear regression).
    **kwargs : TYPE
        Keyword arguments for the matplotlib.pyplot.scatter function. See 
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html.
    
    Returns
    -------
    ax : Axes
        A matplotlib Axes object with the plotted signal & noise data in 
        z-space.

    """
    z_signal = stats.norm.ppf(signal)
    z_noise = stats.norm.ppf(noise)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.axhline(0, lw=1, ls='dashed', c='k')
    ax.axvline(0, lw=1, ls='dashed', c='k')
    
    if data:
        ax.scatter(z_noise, z_signal, **kwargs, zorder=1e10)
    
    if reg:
        thetas = np.polyfit(z_noise, z_signal, poly)
        linex = np.ones(len(z_noise)).reshape(-1, 1)

        # Create polynomial for z(FP)
        for degree in range(1, poly+1):
            linex = np.c_[linex, np.power(z_noise, degree)]
        
        y_pred = linex @ thetas[::-1]
        ax.plot(z_noise, y_pred)

    ax.axis('square')
    ax.set(xlabel='z(FP)', ylabel='z(TP)')
    return ax

def log_likelihood(f_obs, p_exp):
    """Computes the Log Likelihood 
    
    This is the log likelihood function that appears on page 145 of Dunn (2011) 
    and is also used in the ROC toolbox of Koen, Barrett, Harlow, & Yonelinas 
    (2017; see https://github.com/jdkoen/roc_toolbox). The calculation is:
        
        $\sum_i^{j}O_i\log(P_i)$ where $j$ refers the number of response 
        categories.
    
    Parameters
    ----------
    f_obs : TYPE
        The observed frequencies (counts; non-cumulative) for each of the 
        response categories.
    p_exp : TYPE
        The expected probabilities (non-cumulative) for each of the response 
        categories.

    Returns
    -------
    log_likelihood : float
        The log likelihood value for the given inputs.

    """
    
    return (np.array(f_obs) * np.log(np.array(p_exp))).sum()

def squared_errors(observed: np.array, expected: np.array):
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

def aic(k: int, LL: float=None):
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

def bic(k: int, n: int, LL: float):
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

def auc(x: array_like, y: array_like):
    """The area under the curve. In the context of ROC curves, it is equal to 
    the probability that the classifier will be able to discriminate signal 
    from noise.

    Parameters
    ----------
    x : array_like
        The sample points corresponding to the false positive probabilities.
    y : array_like
        The sample points corresponding to the true positive probabilities.

    Returns
    -------
    float or np.ndarray
        The area under the curve. For more details, see 
        https://numpy.org/doc/stable/reference/generated/numpy.trapz.html.

    """
    return np.trapz(x=x, y=y)