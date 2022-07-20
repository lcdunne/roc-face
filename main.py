import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import legacy_chi2 as legacy_stats
from typing import Union, Optional
from matplotlib.axes import Axes

numeric = Union[int, float, np.number]
array_like = Union[list, tuple, np.ndarray]

# Utility funcs. ------------------------------------------------- #
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

def squared_errors(O: np.array, E: np.array):
    """Computes the sum of squared errors between observed values and those 
    which were computed by the model.

    Parameters
    ----------
    O : array_like
        An array of accumulated observed counts.
    E : array_like
        An array of accumulated expected counts.

    Returns
    -------
    np.array
        An array equal to the length of O & E. This contains the computed 
        squared error values for all pairs of error(Oi, Ei). Each element is an 
        estimate of the model fit at the given criterion level i. The sum of 
        these elements is the sum of squared errors.
    """
    return (O - E)**2

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
    TYPE
        DESCRIPTION.

    """
    return 2 * k - 2 * np.log(L)

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

class _BaseModel:
    """Base model class be inherited by all specific model classes. 
    
    Contains functionality and attributes that are used by 
    all models. Not to be instantiated directly.

    Parameters
    ----------
    signal : array_like
        An array of observed response counts to signal-present trials.
    noise : array_like
        An array of observed response counts to noise trials.

    Attributes
    ----------
    

    """
    # Defaults
    __modelname__ = 'none'
    _has_criteria = False
    _named_parameters = {}

    def __init__(self, signal, noise):
        self.shortname = ''.join([i[0] for i in self.__modelname__.split(' ')])
        self.signal = signal
        self.noise = noise
        self.n_signal = sum(self.signal)
        self.n_noise = sum(self.noise)
        self.acc_signal = accumulate(self.signal)
        self.acc_noise = accumulate(self.noise)
        self.p_signal = compute_proportions(self.signal)
        self.p_noise = compute_proportions(self.noise)
        self.auc = auc(x=np.append(self.p_noise, 1), y=np.append(self.p_signal, 1))
        
        if not self._named_parameters:
            _s = {f's{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.p_signal)}
            _n = {f'n{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.p_noise)}
            self._named_parameters = _s | _n
        
        # if getattr(self, '_has_criteria', False):
        if self._has_criteria:
            self.n_criteria = len(self.p_signal)
            self._criteria = {
                f"c{i}": {'initial': 0, 'bounds': (None, None)} for i in range(self.n_criteria)
            }
            self._parameters = self._named_parameters | self._criteria
        else:
            # This will only be the case for high threshold model
            self.n_criteria = 0
            self._parameters = self._named_parameters.copy()
        
        if not hasattr(self, '_n_named_parameters'):
            self._n_named_parameters = len(self._named_parameters)
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__modelname__}>"
    
    @property
    def initial_parameters(self):
        """dict: Starting parameters before applying the fitting procedure."""
        return {k: v['initial'] for k, v in self._parameters.items()}
    
    @property
    def fitted_parameters(self):
        """dict: Parameters after fitting."""
        # TODO: Prevent error when calling this before fitting.
        return self._fitted_parameters
    
    @property
    def parameter_labels(self):
        """list: The labels for all parameters in the model."""
        return list(self._parameters.keys())
    
    @property
    def parameter_boundaries(self):
        """list: The boundary conditions for each parameter during fitting."""
        return list({k: v['bounds'] for k, v in self._parameters.items()}.values())
    
    @property
    def signal_boundary(self):
        """int: The index in the criterion array that corresponds to the 
        boundary between signal and noise (the lowest signal criterion)."""
        if not self._has_criteria:
            return
        c = list(self._criteria.keys())
        return c.index( c[:int(np.ceil(len(c)/2))][-1] )
        
    
    @property
    def n_param(self):
        """int: The number of model parameters."""
        return self._n_named_parameters + self.n_criteria
    
    def define_model_inputs(self, labels: list, values: list, n_criteria: int=0):
        """Maps from flat list of labels and x0 values to dict accepted by the
        `<model>.compute_expected(...)` function.

        Parameters
        ----------
        labels : list
            A list of labels defining the parameter names.
        values : list
            A list of parameter values corresponding to the list of labels. 
            These must be in the same order.
        n_criteria : int, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        dict
            A dictionary that can be unpacked into the modelling function. All 
            named parameters are defined by unique key-value pairs; any 
            criterion parameters are stored in the 'criterion' key, as this 
            parameter in the modelling function requires a list.
        
        Example
        -------
        >>> evsd.define_model_inputs(
                labels=['d', 'c0', 'c1', 'c2', 'c3', 'c4'],
                values=[1.0201, 0.9459, 0.4768, 0.012, -0.5621, -1.2872],
                n_criteria=5
            )
        {'d': 1.0201, 'criteria': [0.9459, 0.4768, 0.012, -0.5621, -1.2872]}

        """
        if n_criteria == 0:
            return dict(zip(labels, values))        
        
        n_named = len(labels) - n_criteria
        
        named_params = dict(zip(labels[:n_named], values[:n_named]))
        criterion_parameters = {'criteria': values[n_named:]}
        
        return named_params | criterion_parameters
    
    def compute_expected(self, *args, **kwargs):
        """Placeholder function for an undefined model.

        Returns
        -------
        model_noise : array_like
            The expected values for the noise array, according to the model.
        model_signal : array_like
            The expected values for the signal array, according to the model.

        """
        return self.p_noise.copy(), self.p_signal.copy()

    def _objective(self, x0: array_like, method: Optional[str]='log-likelihood') -> float:
        """The objective function to minimise. Not intended to be manually 
        called.
        
        During minimization, this function will be called on each iteration 
        with different values of x0 passed in by scipy.stats.optimize.minimize. 
        The values are then passed to the specific theoretical model being 
        fitted, resulting in a set of model-expected values. These expected 
        values are then compared (according to the objective function) with the 
        observed data to produce the resulting value of the objective function.
        
        When calling the <model>.fit() function, the method parameter defines 
        the method of this objective function (e.g. log-likelihood). This 
        argument is passed as a parameter during the call to optimize.minimize.

        Parameters
        ----------
        x0 : array_like
            List of parameters to be optimised. The first call uses the initial 
            guess, corresponding to list(self.initial_parameters). These values 
            are passed to the theoretical model being fitted, and the value of 
            the objective function is then calculated.
        method : str, optional
            See the .fit method for details. The default is 'log-likelihood'.

        Returns
        -------
        float
            The value, for this iteration, of the objective function to be 
            minimized (i.e. a χ^2, G^2, or sum of squared errors).

        """
        # Get the "observed" counts
        observed_signal = self.acc_signal[:-1]
        observed_noise = self.acc_noise[:-1]
        
        # Define the model inputs.
        model_input = self.define_model_inputs(
            labels=self.parameter_labels,
            values=x0, # Not the same as self.x0: this one gets updated
            n_criteria=self.n_criteria
        )
        # Compute the expected probabilities using the model function
        expected_p_noise, expected_p_signal = self.compute_expected(**model_input)
        
        # Compute the expected counts
        # TODO: Make this a function because we want to return this as output
        expected_signal = expected_p_signal * self.n_signal
        expected_noise = expected_p_noise * self.n_noise

        # Compute the fit statistic given observed and model-expected data
        if method == 'log-likelihood':
            # Use the G^2-test
            ll_signal = loglik(O=observed_signal, E=expected_signal, N=self.n_signal)
            ll_noise = loglik(O=observed_noise, E=expected_noise, N=self.n_noise)
            return sum(ll_signal + ll_noise)
        
        elif method == 'chi':
            # Use the chi^2 test
            chi_signal = chitest(observed_signal, expected_signal, self.n_signal)
            chi_noise = chitest(observed_noise, expected_noise, self.n_noise)
            return sum(chi_signal + chi_noise)

        elif method == 'sse':
            # Fit using approach from Yonelinas' spreadsheet
            sse_signal = squared_errors(observed_signal, expected_signal)
            sse_noise = squared_errors(observed_noise, expected_noise)
            return sum(sse_signal + sse_noise)

        elif 'legacy' in method:
            lamb = method.split(' ')[-1] # Gets e.g. 'log-likelihood' (g-test) or 'pearson' (chi-square)
            # Fit using legacy funcs that allow different sums of all counts
            ll_signal, _ = legacy_stats.power_divergence(
                f_obs=[observed_signal, self.n_signal - observed_signal],
                f_exp=[expected_signal, self.n_signal - expected_signal],
                lambda_=lamb
            )
            ll_noise, _ = legacy_stats.power_divergence(
                f_obs=[observed_noise, self.n_noise - observed_noise],
                f_exp=[expected_noise, self.n_noise - expected_noise],
                lambda_=lamb
            )
            return sum(ll_signal) + sum(ll_noise)
    
    def fit(self, method: Optional[str]='log-likelihood'):
        """Fits the theoretical model to the observed data.
        
        Runs the optimisation function according to the chosen method and 
        computes various statistics from the result.
        
        1) Fits the model using scipy.optimize.minimize
        2) Computes the expected signal and noise values after fitting
        3) Computes the squared errors for signal & noise fits
        4) Computes the AIC
        5) Computes euclidean distance between observed and expected values

        Parameters
        ----------
        method : str, optional
            The name of the objective function. Currently accepted values are 
            'log-likelihood', 'legacy log-likelihood', 'legacy pearson', 'chi', 
            and 'sse'. The default is 'log-likelihood'.

        Returns
        -------
        dict
            The fitted parameters.

        """
        # Run the fit function
        self.optimisation_output = minimize(
            fun=self._objective,
            x0=list(self.initial_parameters.values()),
            args=(method),
            bounds=self.parameter_boundaries,
            tol=1e-6
        )
        # Take the results
        self.fitted_values = self.optimisation_output.x

        # Define the model inputs        
        self._fitted_parameters = self.define_model_inputs(
            labels=self.parameter_labels,
            values=self.fitted_values,
            n_criteria=self.n_criteria
        )

        # Compute the expected probabilities using the model function, useful for AIC for example
        self.expected_p_noise, self.expected_p_signal = self.compute_expected(
            **self._fitted_parameters
        )
        
        # TODO: After the above, would be nice to have a method to make all stats
        #   like ._make_results()
        
        # Errors
        # Individual signal & noise are also useful for looking at fits (e.g. plotting "euclidean fit")
        self.signal_squared_errors = (self.p_signal - self.expected_p_signal) ** 2
        self.noise_squared_errors = (self.p_noise - self.expected_p_noise) ** 2
        self.squared_errors = np.concatenate(
            [self.signal_squared_errors, self.noise_squared_errors]
        )
        
        # Compute the AIC
        diffs = np.sqrt(self.squared_errors)
        diffs[diffs == 0] = 1                   # hack 1 (prevent infinite values)
        L = np.product(diffs)**-1               # hack 2 (make it fit to the AIC function)
        self.aic = aic(L=L, k=self.n_param)
        
        # Compute the overall euclidean fit
        signal_euclidean = euclidean_distance(self.p_signal, self.expected_p_signal)
        noise_euclidean = euclidean_distance(self.p_noise, self.expected_p_noise)
        self.euclidean_fit = signal_euclidean + noise_euclidean
        
        # TODO: Define nice results output
        self.results = {
            'model': self.__modelname__,
            'opt-success': self.optimisation_output.success,
            method: self.optimisation_output.fun,
            'aic': self.aic,
            'euclidean_fit': self.euclidean_fit,
        }
        
        return self.fitted_parameters


class HighThreshold(_BaseModel):
    """High Threshold model class. Inherits functionality from _BaseModel class.
    
    See Yonelinas et al. (1996).

    Parameters
    ----------
    signal : array_like
        An array of observed response counts to signal-present trials.
    noise : array_like
        An array of observed response counts to noise trials.

    Attributes
    ----------
    https://softwareengineering.stackexchange.com/questions/353004/how-should-i-handle-docstrings-of-subclass-methods

    """
    __modelname__ = 'High Threshold'
    _has_criteria = False

    def __init__(self, signal, noise):
        self._named_parameters = {'R': {'initial': 0.999, 'bounds': (0, 1)}}
        self._n_named_parameters = len(self._named_parameters) + 1 # Required because `g` parameter is implicit
        self.label = ''.join([i[0] for i in self.__modelname__.split()])
        super().__init__(signal, noise)
    
    def compute_expected(self, R: float, full: Optional[bool]=False) -> tuple:
        """Compute the expected signal and noise array using the High Threshold 
        model.

        Parameters
        ----------
        R : float
            Threshold parameter, corresponding to the probability of 
            recollection - the only variable to be solved for in the 
            High Threshold model.
        full : bool, optional
            Whether to extend the model line across probability space. The 
            default is False.
        
        Returns
        -------
        model_noise : array_like
            The expected values for the noise array, according to the model.
        model_signal : array_like
            The expected values for the signal array, according to the model.

        """
        if full:
            model_noise = np.array([0, 1])
        else:
            model_noise = self.p_noise
        model_signal = (1 - R) * model_noise + R
        return model_noise, model_signal


class SignalDetection(_BaseModel):
    """Signal Detection model class. Inherits functionality from _BaseModel 
    class.
    
    See Wixted.

    Parameters
    ----------
    signal : array_like
        An array of observed response counts to signal-present trials.
    noise : array_like
        An array of observed response counts to noise trials.
    equal_variance: bool, Optional
        Whether the variance of the signal distribution should be equal to that 
        of the noise distribution, or not. If not, then the signal variance is 
        considered as an additional parameter to be solved for.

    Attributes
    ----------
    https://softwareengineering.stackexchange.com/questions/353004/how-should-i-handle-docstrings-of-subclass-methods

    """
    __modelname__ = 'Equal Variance Signal Detection'
    _has_criteria = True

    def __init__(self, signal, noise, equal_variance=True):
        self._named_parameters = {'d': {'initial': 0, 'bounds': (None, None)}}
        
        if not equal_variance:
            self.__modelname__ = self.__modelname__.replace('Equal', 'Unequal')
            self._named_parameters['scale'] = {'initial': 1, 'bounds': (1, None)}
        
        self._scale = 1.0 # Define the scale of the signal distribution
        
        self.label = ''.join([i[0] for i in self.__modelname__.split()])
        super().__init__(signal, noise)
    
    @property
    def scale(self):
        # TODO: would be better as self.fitted_parameters.get('scale', 1)
        # with default fitted_parameters as an empty dict on init.
        return self._scale

    def compute_expected(
            self,
            d: float,
            scale: Optional[float]=1,
            criteria: Optional[array_like]=None
        ) -> tuple:
        """Compute the expected signal and noise array using the Signal Detection 
        model.

        Parameters
        ----------
        d : float
            Sensitivity parameter. Corresponds to the distance between the 
            signal and noise distributions.
        scale : float, optional
            The standard deviation of the signal distribution. The default is 1.
        criteria : array_like, optional
            Criterion parameter values. The length corresponds to the number of
            response categories minus 1 which are solved for. The 
            default is None.

        Returns
        -------
        model_noise : array_like
            The expected values for the noise array, according to the model.
        model_signal : array_like
            The expected values for the signal array, according to the model.

        """
        if criteria is None:
            criteria = np.arange(-5, 5, 0.01)
        
        self._scale = scale

        model_signal = stats.norm.cdf(d / 2 - np.array(criteria), scale=scale)
        model_noise = stats.norm.cdf(-d / 2 - np.array(criteria), scale=1)
        
        return model_noise, model_signal


class DualProcess(_BaseModel):
    """Dual Process model class. Inherits functionality from _BaseModel 
    class.
    
    This is a combination of the equal-variance signal detection and the high 
    threshold models.

    Parameters
    ----------
    signal : array_like
        An array of observed response counts to signal-present trials.
    noise : array_like
        An array of observed response counts to noise trials.

    Attributes
    ----------
    https://softwareengineering.stackexchange.com/questions/353004/how-should-i-handle-docstrings-of-subclass-methods

    """
    __modelname__ = 'Dual Process Signal Detection'
    _has_criteria = True

    def __init__(self, signal, noise):
        self._named_parameters = {
            'd': {'initial': 0, 'bounds': (None, None)},
            'R': {'initial': 0.999, 'bounds': (0, 1)},
        }
        self._recollection = None
        self._familiarity = None
        
        self.label = ''.join([i[0] for i in self.__modelname__.split()])
        super().__init__(signal, noise)
    
    @property
    def familiarity(self):
        """float: Estimate of familiarity."""
        return self._familiarity
    
    @property
    def recollection(self):
        """float: Estimate of recollection."""
        return self._recollection    

    def compute_expected(
            self,
            d: float,
            R: float,
            criteria: Optional[array_like]=None
        ) -> tuple:
        """Compute the expected signal and noise array using the Dual Process 
        model.
        
        See Yonelinas (1996).

        Parameters
        ----------
        d : float
            Sensitivity parameter. Corresponds to the distance between the 
            signal and noise distributions. Viewed as an index of familiarity 
            under the dual-process model.
        R : float
            Threshold parameter, corresponding to the probability of 
            recollection.
        criteria : array_like, optional
            Criterion parameter values. The length corresponds to the number of
            response categories minus 1 which are solved for. The 
            default is None.

        Returns
        -------
        model_noise : array_like
            The expected values for the noise array, according to the model.
        model_signal : array_like
            The expected values for the signal array, according to the model.

        """
        if criteria is None:
            criteria = np.arange(-5, 5, 0.01)
        
        # Estimate familiarity & recollection
        c = list(evsd._criteria.keys())
        c[:int(np.ceil(len(c)/2))][-1]
        
        # signal_boundary = criteria[: int( np.ceil( len( criteria ) / 2 ))][-1]
        self._familiarity = stats.norm.cdf( d / 2 - criteria[self.signal_boundary] )
        self._recollection = R

        model_noise = stats.norm.cdf(-d / 2 - criteria)
        model_signal = R + (1 - R) * stats.norm.cdf(d / 2 - criteria)
        return model_noise, model_signal


if __name__ == '__main__':
    
    signal = [505,248,226,172,144,93]
    noise = [115,185,304,523,551,397]
    
    ht = HighThreshold(signal, noise)
    ht.fit()
    print(ht.results)
    
    evsd = SignalDetection(signal, noise, equal_variance=True)
    evsd.fit()
    print(evsd.results)
    
    uvsd = SignalDetection(signal, noise, equal_variance=False)
    uvsd.fit()
    print(uvsd.results)

    dpsd = DualProcess(signal, noise)
    dpsd.fit()
    print(dpsd.results)
    
    # Plot
    fig, ax = plt.subplots(dpi=150)

    plot_roc(ht.p_signal, ht.p_noise, ax=ax)
    
    ax.plot(*ht.compute_expected(**ht.fitted_parameters), label=ht.label)
    ax.plot(*evsd.compute_expected(**evsd.fitted_parameters), label=evsd.label)
    ax.plot(*uvsd.compute_expected(**uvsd.fitted_parameters), label=uvsd.label)
    ax.plot(*dpsd.compute_expected(**dpsd.fitted_parameters), label=dpsd.label)

    ax.legend(loc='lower right')
    plt.show()
    
    fig, ax = plt.subplots(1, 3, figsize=(9,4), dpi=100, sharey=True)
    ax[0].bar(x=np.arange(1,11), height=ht.squared_errors)
    ax[1].bar(x=np.arange(1,11), height=evsd.squared_errors)
    ax[2].bar(x=np.arange(1,11), height=uvsd.squared_errors)
    ax[0].set(ylabel='Log Euclidean Fit', xlabel='criterion', yscale='log', title='High Threshold')
    ax[1].set(title='Equal Variance', yscale='log', xlabel='criterion',)
    ax[2].set(title='Unequal Variance', yscale='log', xlabel='criterion',)
    plt.show()