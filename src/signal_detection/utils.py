import numpy as np
from numpy.typing import ArrayLike
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats
from typing import Union, Optional, List

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
    corrected: bool=True,
    truncate: bool=True
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
        from_freqs: bool=True,
        ax: Optional[Axes]=None,
        chance: bool=True,
        **kwargs
    ) -> Axes:
    """A utility to plot ROC curves. Requires signal and noise arrays in 
    probability space. Accepts scatter plot keyword arguments.
    
    Parameters
    ----------
    signal : array_like
        Signal array of responses. When `from_freqs=True` (the default), the 
        values are assumed to be the raw observed frequencies for each response
        category.
    noise : array_like
        array of responses. When `from_freqs=True` (the default), the 
        values are assumed to be the raw observed frequencies for each response
        category.
    from_freqs: True, optional
        Specifies whether the arguments to `signal` and `noise` contain the 
        raw frequency data (`from_freqs=True`) or if they are already prepared 
        and in ROC space (cumulative probabilities; `from_freqs=False`). The 
        default is True.
    ax : Optional[Axes], optional
        Matplotlib Axes object to plot to, if already defined. The default is 
        None.
    chance : bool, optional
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
    if from_freqs:
        signal = compute_proportions(signal)
        noise = compute_proportions(noise)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if chance:
        ax.plot([0,1], [0,1], c='k', lw=1, ls='dashed')
    
    ax.scatter(noise, signal, **kwargs, zorder=1e10)
    ax.axis('square')
    ax.set(xlabel='FP', ylabel='TP')
    return ax

def regress(x: array_like, y: array_like, poly: int=1) -> tuple:
    """Simple regression with optional polynomial expansion.

    Parameters
    ----------
    x : array_like
        The set of x values (predictor).
    y : array_like
        The set of y values (outcome).
    poly : int, optional
        The order of the polynomial regression. The 
        default is 1 (linear regression).

    Returns
    -------
    coefs : array_like
        Parameters of the regression, obtained using numpy.polyfit (see the 
        NumPy docs), in descending order from highest degree to lowest, with 
        element `coefs[-1]` being the intercept of the line.
    y_pred : array_like
        The predictions for y given the parameters.

    """
    coefs = np.polyfit(x, y, poly)
    x_ = np.ones(len(x)).reshape(-1, 1)
    
    for degree in range(1, poly+1):
        x_ = np.c_[x_, np.power(x, degree)]
    
    y_pred = x_ @ coefs[::-1]
    
    return coefs, y_pred

def linear_equation(coefs: Union[float, array_like], precision: int=2):
    """Generate a string representation of a polynomial regression equation.

    Parameters
    ----------
    coefs : array_like
        An array of regression coefficients, highest power first (see the docs 
        for `numpy.polyfit` return values).
    
    Examples
    --------
    >>> linear_equation([.69])       
    'y = 0.69'

    >>> linear_equation([.42, .69])                         
    'y = 0.69 + 0.42x'

    >>> linear_equation([1.4195, -0.89324, .42013, .69069], precision=4) 
    'y = 0.6907 + 0.4201x - 0.8932x^2 + 1.4195x^3'

    Returns
    -------
    equation : str
        The string representation of an equation.
    """
    if isinstance(coefs, float):
        coefs = [coefs]
    
    intercept = np.round(coefs[-1], precision)

    equation = f'y = {intercept} + '

    if len(coefs) == 1:
        return equation.replace(' + ', '')
    coefs = coefs[:-1][::-1]  
    
    equation += ' + '.join([f'{np.round(coef, precision)}x^{i+1}' for i, coef in enumerate(coefs)])
    equation = equation.replace('+ -', '- ')
    equation = equation.replace('^1', '')
    return equation

def plot_zroc(
        signal: array_like,
        noise: array_like,
        from_freqs: bool=True,
        reg: bool=True,
        poly: int=1,
        data: bool=True,
        show_equation: bool=True,
        ax: Optional[Axes]=None,
        scatter_kwargs: dict={},
        line_kwargs: dict={}

    ):
    """A utility to plot z-ROC curves. Requires signal and noise arrays in 
    probability space. Accepts scatter plot keyword arguments.
    
    Parameters
    ----------
    signal : array_like
        Signal array of responses. When `from_freqs=True` (the default), the 
        values are assumed to be the raw observed frequencies for each response
        category.
    noise : array_like
        array of responses. When `from_freqs=True` (the default), the 
        values are assumed to be the raw observed frequencies for each response
        category.
    from_freqs: True, optional
        Specifies whether the arguments to `signal` and `noise` contain the 
        raw frequency data (`from_freqs=True`) or if they are already prepared 
        and in z-ROC space (z-scores of the cumulative probabilities; 
        `from_freqs=False`). The default is True.
    ax : Optional[Axes], optional
        Matplotlib Axes object to plot to, if already defined. The default is 
        None.
    reg : bool, optional
        Whether or not to draw a regression line. If True, see `poly`. The 
        default is True.
    poly : Optional[int], optional
        The order of the polynomial regression line. The 
        default is 1 (linear regression).
    show_equation : bool, optional
        Whether to show the equation as the label of the line (if plotted).
    scatter_kwargs : dict, optional
        Keyword arguments for the matplotlib.pyplot.scatter function. See 
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html.
    line_kwargs : dict, optional
        Keyword arguments for the matplotlib.pyplot.plot function. See 
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html.
    
    Returns
    -------
    ax : Axes
        A matplotlib Axes object with the plotted signal & noise data in 
        z-space.

    """
    if from_freqs:
        signal = compute_proportions(signal)
        noise = compute_proportions(noise)
    
    z_signal = stats.norm.ppf(signal)
    z_noise = stats.norm.ppf(noise)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.axhline(0, lw=1, ls='dashed', c='k')
    ax.axvline(0, lw=1, ls='dashed', c='k')
    
    if data:
        ax.scatter(z_noise, z_signal, zorder=1e10, **scatter_kwargs)

    if reg:
        coefs, y_pred = regress(x=z_noise, y=z_signal, poly=poly)
        
        if show_equation and 'label' not in line_kwargs:
            # Only show the equation if requested and if label not already provided.
            line_kwargs['label'] = "$" + linear_equation(coefs) + "$"
        ax.plot(z_noise, y_pred, **line_kwargs)

    ax.axis('square')
    ax.set(xlabel='z(FP)', ylabel='z(TP)')
    return ax



def auc(x: array_like, y: array_like) -> float:
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

def plot_performance(
        dprime: float,
        scale: float=1,
        ax: Optional[Axes]=None,
        shade: bool=False,
        noise_kwargs: dict={'c': 'k'},
        signal_kwargs: dict={}
    ):
    """Plot a signal & noise distribution for a specific value of d`.
    
    Parameters
    ----------
    dprime : float
        Sensitivity index.
    scale : float, optional
        The standard deviation of the signal distribution. The default is 1.
    ax : Optional[Axes], optional
        Matplotlib Axes object to plot to, if already defined. The default is 
        None.
    shade : bool, optional
        Whether to shade the distributions. The default is False.
    noise_kwargs : dict, optional
        Keyword arguments (see matplotlib.pyplot.plot) for the noise 
        distribution. The default is {'c': 'k'}.
    signal_kwargs : dict, optional
        Keyword arguments (see matplotlib.pyplot.plot) for the signal 
        distribution. The default is {}.

    Returns
    -------
    ax : Axes
        A matplotlib Axes object displaying the signal & noise distributions 
        for the specified value of d`.

    """
    strength = np.linspace(-4, 4+dprime*scale, 1000)    
    noisedist = stats.norm.pdf(strength, loc=0, scale=1)
    signaldist = stats.norm.pdf(strength, loc=dprime, scale=scale)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(strength, noisedist, label='noise', **noise_kwargs)
    noise_clr = plt.gca().lines[-1].get_color()
    ax.plot(strength, signaldist, label='signal', **signal_kwargs)
    signal_clr = plt.gca().lines[-1].get_color()
    plt.gca().lines[-1].get_linewidth()
    
    if shade:
        ax.fill_between(strength, noisedist, color=noise_clr, alpha=1/3)
        ax.fill_between(strength, signaldist, color=signal_clr, alpha=1/3)
    
    ax.set(yticklabels=[], yticks=[])
    return ax
