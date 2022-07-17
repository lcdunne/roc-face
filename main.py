import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import legacy_chi2 as legacy_stats

# Utility funcs. ------------------------------------------------- #
def array_like(x):
    return isinstance(x, (list, tuple, np.ndarray,))

def arrays_equal_length(a, b):
    if len(a) != len(b):
        return False
    else:
        return True

def accumulate(arr):
    if not array_like(arr):
        raise ValueError("Expected array like")
    return np.cumsum(arr)

def corrected_proportion(a, x, i=0):
    return (x + i / len(a)) / (max(a) + 1)

def compute_proportions(arr, truncate=True):
    """Accumulates an array and converts it to proportions of its total."""
    # Corrects for ceiling effects.
    a = accumulate(arr)
    freq = [corrected_proportion(a, x, i) for i, x in enumerate(a, start=1)]
    
    if freq[-1] != 1:
        raise ValueError(f"Expected max accumulated to be 1 but got {freq[-1]}.")
    
    if truncate:
        freq.pop()

    return np.array(freq)

def plot_roc(signal, noise, ax=None, chance=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    
    if chance:
        ax.plot([0,1], [0,1], c='k', lw=1, ls='dashed')
    
    ax.scatter(noise, signal, **kwargs)
    ax.axis('square')
    ax.set(xlabel='1 - specificity', ylabel='sensitivity')
    return ax

# Fitting functions
def loglik(O, E, N):
    # G-test (https://en.wikipedia.org/wiki/G-test)
    with np.errstate(divide='ignore'):
        # ignore infinite value warning & return inf anyway.
        return 2 * O * np.log(O/E) + 2 * (N - O) * np.log((N - O)/(N - E))

def chitest(O, E, N):
    return (O - E)**2 / E + ((N-O) - (N-E))**2 / (N-E)

def sum_sse(O, E):
    return sum( (O - E)**2 )

def aic(L, k):
    # k: number of estimated parameters in the model
    return 2 * k - 2 * np.log(L)


class BaseModel:
    __modelname__ = 'none'

    def __init__(self, signal, noise):
        '''
        Model instantiation will create...
        Why not just use a dict with "criteria" as a key and an ordered array...?

        Parameters
        ----------
        signal : TYPE
            DESCRIPTION.
        noise : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.shortname = ''.join([i[0] for i in self.__modelname__.split(' ')])
        self.signal = signal
        self.noise = noise
        self.n_signal = sum(self.signal)
        self.n_noise = sum(self.noise)
        self.acc_signal = accumulate(self.signal)
        self.acc_noise = accumulate(self.noise)
        self.p_signal = compute_proportions(self.signal)
        self.p_noise = compute_proportions(self.noise)
        
        # Since multiple models will have criteria, best to init them in this parent class
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
    
    @property
    def initial_parameters(self):
        return {k: v['initial'] for k, v in self._parameters.items()}
    
    @property
    def fitted_parameters(self):
        # return {k: v.get('fitted') for k, v in self._parameters.items() if v.get('fitted') is not None}
        return self._fitted_parameters
    
    @property
    def parameter_labels(self):
        return list(self._parameters.keys())
    
    @property
    def parameter_boundaries(self):
        return list({k: v['bounds'] for k, v in self._parameters.items()}.values())
    
    @property
    def n_param(self):
        return len(self.parameter_labels)
    
    @property
    def initial_input(self):
        # This is not even a necessary variable - just use it directly as input
        return list(self.initial_parameters.values())
    
    def define_model_inputs(self, labels, values, n_criteria=0):
        # Maps from flat list of labels and x0 values to dict accepted by `<model>.compute_expected`
        if n_criteria == 0:
            return dict(zip(labels, values))        
        # Number of named i.e. non-criteria parameters
        n_named = len(labels) - n_criteria
        return dict(zip(labels[:n_named], values[:n_named])) | {'criteria': values[n_named:]}

    def subset_dict(self, d, withkeys=None):
        if withkeys is None:
            return {k: v for k, v in d.items()}
        return {k: v for k, v in d.items() if k in withkeys}

    def objective(self, x0, method='log-likelihood'):
        '''
        Uses parameters defined in x0 to pass to the theoretical model and compute expected values.
        
        These expected values are then assessed in terms of their fit according to the specified method.
        
        Method must be one of:
            'log-likelihood'
            'legacy log-likelihood'
            'legacy pearson'
            'sse'
        
        The result of this fit procedure is a single statistic.
        
        The minimization algorithm updates the x0 params until the fit statistic is been minimized.
        '''
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
            sse_signal = sum_sse(observed_signal, expected_signal)
            sse_noise = sum_sse(observed_noise, expected_noise)
            return sse_signal + sse_noise

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
    
    def fit(self, method='log-likelihood'):
        '''
        To return:
            {method: result, success: bool, observed_prob, expected_prob, observed_count, expected_count}
        '''
        # Run the fit function
        self.optimisation_output = minimize(
            fun=self.objective,
            x0=self.initial_input,
            args=(method),
            bounds=self.parameter_boundaries,
            tol=1e-6
        )
        # Take the results
        self.x0 = self.optimisation_output.x

        # Define the model inputs        
        self._fitted_parameters = self.define_model_inputs(
            labels=self.parameter_labels,
            values=self.x0,
            n_criteria=self.n_criteria
        )

        # Compute the expected probabilities using the model function, useful for AIC for example
        self.expected_p_noise, self.expected_p_signal = self.compute_expected(
            **self._fitted_parameters
        )
        
        # Errors
        self._signal_squared_errors = (self.p_signal - self.expected_p_signal) ** 2
        self._noise_squared_errors = (self.p_noise - self.expected_p_noise) ** 2
        self.squared_errors = np.concatenate(
            [self._signal_squared_errors, self._noise_squared_errors]
        )
        
        if any(self.squared_errors==0):
            self.squared_errors = self.squared_errors[self.squared_errors != 0]
        self.aic = aic(L=sum(-np.log(np.sqrt(self.squared_errors))), k=self.n_param)
        
        
        
        # TODO: Define nice results output
        
        return self._fitted_parameters
    
    def euclidean_misfit(self, ox, oy, ex, ey):
        # TODO
        pass


class HighThreshold(BaseModel):
    __modelname__ = 'High Threshold'
    _has_criteria = False

    def __init__(self, signal, noise):
        self._named_parameters = {'R': {'initial': 0.999, 'bounds': (0, 1)}}
        # TODO: This model actually has 2 parameters, with `g` (guess). Currently self.n_param returns just 1. would be good to fix the compute_expected for this.
        super().__init__(signal, noise)
    
    def __repr__(self):
        return f"<{sdt.__class__.__name__}: {self.__modelname__}>"
    
    def compute_expected(self, R, full=False):
        if full:
            model_noise = np.array([0, 1])
        else:
            model_noise = self.p_noise
        model_signal = (1 - R) * model_noise + R
        return model_noise, model_signal


class SignalDetection(BaseModel):
    __modelname__ = 'Equal Variance Signal Detection'
    _has_criteria = True

    def __init__(self, signal, noise, equal_variance=True):
        self._named_parameters = {'d': {'initial': 0, 'bounds': (None, None)}}
        
        if not equal_variance:
            self.__modelname__ = self.__modelname__.replace('Equal', 'Unequal')
            self._named_parameters['scale'] = {'initial': 1, 'bounds': (1, None)}

        self.label = ''.join([i[0] for i in self.__modelname__.split()])
        super().__init__(signal, noise)
    
    def __repr__(self):
        return f"<{sdt.__class__.__name__}: {self.__modelname__}>"

    # SDT function to get f_exp
    def compute_expected(self, d=None, scale=1, criteria=None):
        if criteria is None:
            criteria = np.arange(-5, 5, 0.01)

        model_signal = stats.norm.cdf(d / 2 - np.array(criteria), scale=scale)
        model_noise = stats.norm.cdf(-d / 2 - np.array(criteria), scale=1)
        return model_noise, model_signal

if __name__ == '__main__':
    signal = [505,248,226,172,144,93]
    noise = [115,185,304,523,551,397]
    sdt = SignalDetection(signal, noise, equal_variance=True)
    sdt.fit()
    print(sdt.optimisation_output)

    ht = HighThreshold(signal, noise)
    ht.fit()
    print(ht.optimisation_output)