import numpy as np
from scipy.stats import norm
from typing import Union, Optional
zscore = norm.ppf

def compute_performance(tpr: float, fpr: float, z_intercept: Optional[float]=None, z_slope: Optional[float]=None) -> dict:
    performance = {
            'TPR': tpr,
            'FPR': fpr,
            'dprime': d_prime(tpr, fpr),
            'aprime': a_prime(tpr, fpr),
            'cbias': c_bias(tpr, fpr),
            'beta': beta(tpr, fpr),
    }

    if z_intercept is not None and z_slope is not None:
        performance['Az'] = a_z(z_intercept, z_slope)
    
    return performance
        


def d_prime(tpr: Union[float, np.ndarray], fpr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sensitivity measure d\`, element-wise.

    Parameters
    ----------
    tpr : array_like
        The true positive rate(s) (range 0 <= `tpr` <= 1)
    fpr : array_like
        The false positive rate(s) (range 0 <= `fpr` <= 1)

    Returns
    -------
    ndarray
        d\`. An array of the same length as tpr, fpr, containing the d\`
        values. If `tpr` and `fpr` are single floats then a single float
        corresponding to d\` is returned.
    
    Notes
    -----
    Estimates the ability to discriminate between instances of signal and
    noise. The larger the magnitude, the greater the sensitivity.
    
    Example
    -------
    >>> d = d_prime(0.75, 0.21)
    >>> print(d)
    1.480910997214322
    """
    return zscore(tpr) - zscore(fpr)


def c_bias(tpr: Union[float, np.ndarray], fpr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Bias measure `c`.

    Parameters
    ----------
    tpr : array_like
        The true positive rate(s) (range 0 <= `tpr` <= 1). Can be
        a single float value or an array.
    fpr : array_like
        The false positive rate(s) (range 0 <= `fpr` <= 1). Can be
        a single float value or an array.

    Returns
    -------
    ndarray
        c. An array of the same length as tpr, fpr, containing the c values.
        If `tpr` and `fpr` are single floats then a single float corresponding
        to c is returned.
    
    Notes
    -----
    Estimates the criterion value relative to the intersection of the signal
    and noise distributions, centering on 0 (no bias). Positive values 
    indicate conservative bias, while negative values indicate a liberal
    bias.

    Example
    -------
    >>> c = c_bias(0.75, 0.21)
    >>> print(c)
    0.06596574841107933
    """
    return 1 / 2 * -(zscore(tpr) + zscore(fpr))


def a_prime(tpr: float, fpr:float) -> float:
    """The sensitivity index `A\``.

    Parameters
    ----------
    tpr : array_like
        The true positive rate(s) (range 0 <= `tpr` <= 1). Can be
        a single float value or an array.
    fpr : array_like
        The false positive rate(s) (range 0 <= `fpr` <= 1). Can be
        a single float value or an array.

    Returns
    -------
    ndarray
        A\`. An array of the same length as tpr, fpr, containing the A\`
        values. If `tpr` and `fpr` are single floats then a single float
        corresponding to A\` is returned.
    
    Notes
    -----
    A non-parametric measure of discrimination that estimates
    sensitivity as the area under an "average" ROC curve, for a
    single true- and false-positive rate. For further details,
    see Snograss & Corwin (1988). Note that `d\`` is  preferred.
    """
    if tpr >= fpr:
        numerator = ((tpr - fpr) * (1 + tpr - fpr))
        denominator = 4 * tpr * (1 - fpr)
        return 0.5 + (numerator / denominator)
    else:
        numerator = ((fpr - tpr) * (1 + fpr - tpr))
        denominator = 4 * fpr * (1 - tpr)
        return 0.5 - (numerator / denominator)


def beta(tpr: Union[float, np.ndarray], fpr: Union[float, np.ndarray]):
    """The bias measure, β.

    Parameters
    ----------
    tpr : array_like
        The true positive rate(s) (range 0 <= `tpr` <= 1). Can be
        a single float value or an array.
    fpr : array_like
        The false positive rate(s) (range 0 <= `fpr` <= 1). Can be
        a single float value or an array.

    Returns
    -------
    ndarray
        β. An array of the same length as tpr, fpr, containing the β values.
        If `tpr` and `fpr` are single floats then a single float corresponding
        to β is returned.
    
    Notes
    -----
    β is a likelihood ratio measure that estimates the criterion
    value by computing the ratio of the heights of the signal &
    noise distributions. It is a ratio of the density of the signal
    distribution at the criterion divided by the same density for
    the noise distribution.

    It is the ratio of the likelihood of obtaining an observation
    equal to the criterion given a signal to the likelihood of
    obtaining this observation given noise. Or: L(x_c)|S / L(x_c)|N.

    See Snodgrass & Corwin (1988) for details.
    """
    return np.exp( (zscore(fpr)**2 -  zscore(tpr)**2) / 2 )


def beta_doubleprime(tpr: Union[float, np.ndarray], fpr: Union[float, np.ndarray], donaldson: bool=False) -> Union[float, np.ndarray]:
    """The bias index `β\`\``.

    Parameters
    ----------
    tpr : array_like
        The true positive rate(s) (range 0 <= `tpr` <= 1). Can be
        a single float value or an array.
    fpr : array_like
        The false positive rate(s) (range 0 <= `fpr` <= 1). Can be
        a single float value or an array.
    donaldson : bool, optional
        Use Donaldson's calculation, by default False. If this value is
        `False`, use Grier's (1971) as cited in Stanislaw & Todorov (1999).

    Returns
    -------
    float, np.ndarray
        β\`\`. An array of the same length as tpr, fpr, containing the β\`\`
        values. If `tpr` and `fpr` are single floats then a single float
        corresponding to β\`\` is returned.
    
    Notes
    -----
    Grier's `β\`\`` is A2-A1 divided by sum(A1, A2). If A1 < A2,
    there is a liberal bias, and if A2 < A1, there is a conservative
    bias.
    
    Default is Grier's (1971) as cited in Stanislaw & Todorov (1999), but 
    can be changed to use Donaldson's calculation.
    """
    fnr = 1 - tpr
    tnr = 1 - fpr

    if donaldson:
        numerator = (fnr * tnr) - (tpr * fpr)
        denominator = (fnr * tnr) + (tpr * fpr)
        return numerator / denominator

    else:
        numerator = (tpr * fnr) - (fpr * tnr)
        denominator = (tpr * fnr) + (fpr * tnr)
        return np.sign(tpr - fpr) * numerator / denominator

def a_z(z_intercept: float, z_slope: float) -> float:
    # See Stanislaw & Todorov (1999). Useful for checking the d` assumption of
    # equal variances: slope should = 1 (or log(slope) == 0).
    return norm.cdf( z_intercept / np.sqrt( 1 + z_slope**2 ) )

if __name__ == '__main__':
    print(d_prime(.75, .21))
    print(c_bias(.75, .21))
    print(a_prime(.75, .21))
    print(beta(.75, .21))
    print(beta_doubleprime(.75, .21))
    print(beta_doubleprime(.75, .21, donaldson=True))
    print(compute_performance(.75, .21, 1.23, 0.79))

