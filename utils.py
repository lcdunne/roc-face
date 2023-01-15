import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Union, Optional

numeric = Union[int, float, np.number]
array_like = Union[list, tuple, np.ndarray]

def arrays_equal_length(a: array_like, b: array_like):
    if len(a) != len(b):
        return False
    else:
        return True

def accumulate(arr: array_like):
    return np.cumsum(arr)

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

def euclidean_distance(x: np.array, y: np.array):
    return np.sqrt(sum((y - x)**2))

def plot_roc(
        signal: array_like,
        noise: array_like,
        ax: Optional[Axes]=None,
        chance: Optional[bool]=True,
        **kwargs
    ) -> Axes:
    """A utility to plot ROC curves. Requires signal and noise arrays in 
    probability space. Accept scatter plot keyword arguments.
    
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
        A matplotlib Axes object with the plotted signal & noise data.

    """
    if ax is None:
        fig, ax = plt.subplots()
    
    if chance:
        ax.plot([0,1], [0,1], c='k', lw=1, ls='dashed')
    
    ax.scatter(noise, signal, **kwargs)
    ax.axis('square')
    ax.set(xlabel='1 - specificity', ylabel='sensitivity')
    return ax

# de-accumulate
def deaccumulate(arr: array_like) -> np.ndarray:
    return np.diff(np.insert(arr, 0, 0)) # insert a 0 at the start

# Fitting functions
def loglik(O: np.array, E: np.array, N: numeric):
    """Computes the G-test (https://en.wikipedia.org/wiki/G-test).
    Note that this function is equivalent to 
    `scipy.stats.power_divergence(f_obs, f_exp, ... lambda_='log-likelihood')`.

    Parameters
    ----------
    O : array_like
        An array of accumulated observed counts.
    E : array_like
        An array of accumulated expected counts.
    N : numeric
        The total number of responses for the set. As currently implemented,
        O is truncated and will not contain the total N at O[-1], so N must be 
        passed explicitly. This may be changed in a future implementation.

    Returns
    -------
    np.array
        An array equal to the length of O & E. This contains the computed G^2 
        values for all pairs of G(Oi, Ei). Each element is an estimate of the 
        model fit at the given criterion level i. The sum of these elements is 
        the sum of G^2, which can then be further analysed.
    """
    # TODO: N could be obtained with N = O[-1] if we do not truncate the input.
    #   however this would mean len(O) == len(E)+1 which is a little clunky.
    with np.errstate(divide='ignore'):
        # ignore infinite value warning & return inf anyway.
        # alternative return could be just the sum of this.
        return 2 * O * np.log(O/E) + 2 * (N - O) * np.log((N - O)/(N - E))

def chitest(O: np.array, E: np.array, N: numeric):
    """Computes Pearson's χ^2 test (https://en.wikipedia.org/wiki/Chi-squared_test). 
    Note that this function is equivalent to 
    `scipy.stats.power_divergence(f_obs, f_exp, ... lambda_='pearson')`.

    Parameters
    ----------
    O : array_like
        An array of accumulated observed counts.
    E : array_like
        An array of accumulated expected counts.
    N : numeric
        The total number of responses for the set. As currently implemented,
        O is truncated and will not contain the total N at O[-1], so N must be 
        passed explicitly. This may be changed in a future implementation.

    Returns
    -------
    np.array
        An array equal to the length of O & E. This contains the computed χ^2 
        values for all pairs of χ^2(Oi, Ei). Each element is an estimate of the 
        model fit at the given criterion level i. The sum of these elements is 
        the sum of χ^2, which can then be further analysed.
    """
    return (O - E)**2 / E + ((N-O) - (N-E))**2 / (N-E)

def squared_errors(p_o: np.array, p_e: np.array):
    """Computes the sum of squared errors between observed values and those 
    which were computed by the model.

    Parameters
    ----------
    p_o : array_like
        Array of (accumulated) observed probabilities.
    p_e : array_like
        Array of (accumulated) expected probabilities (from the model).

    Returns
    -------
    np.array
        An array equal to the length of p_o & p_e. This contains the computed 
        squared error values for all pairs along the curve. Each element is an 
        estimate of the model fit at the given criterion point. The sum of 
        these elements is the sum of squared errors.
    """
    return (p_o - p_e)**2

def aic(L: float, k: int):
    """Computes Akaike's information criterion (AIC; https://en.wikipedia.org/wiki/Akaike_information_criterion).
    
    Is an estimator of quality of each model relative to others, enabling model 
    comparison and selection.

    Parameters
    ----------
    L : float
        The maximum value of the likelihood function for the model. In the 
        present context, this is the sum of the negative log of the errors of 
        a given model, i.e. sum(-ln(sqrt(squared_errors))).
    k : int
        The number of estimated parameters in the model.

    Returns
    -------
    float
        The AIC score.

    """
    return 2 * k - 2 * np.log(L)

def bic(L: float, k: int, n: int):
    """Computes the Bayesian information criterion (BIC; https://en.wikipedia.org/wiki/Bayesian_information_criterion).
    
    Is an estimator of quality of each model relative to others, enabling model 
    comparison and selection.

    Parameters
    ----------
    L : float
        The maximum value of the likelihood function for the model. In the 
        present context, this is the sum of the negative log of the errors of 
        a given model, i.e. sum(-ln(sqrt(squared_errors))).
    k : int
        The number of estimated parameters in the model.
    n : int
        The number of data points in the observed data.

    Returns
    -------
    float
        The BIC score.

    """
    return k * np.log(n) - 2 * np.log(L)




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