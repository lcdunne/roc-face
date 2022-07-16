import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import legacy_chi2 as legacy_stats

# Utility funcs. ------------------------------------------------- #
def accumulate(arr):
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

def arrays_equal_length(a, b):
    if len(a) != len(b):
        return False
    else:
        return True

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
    with np.errstate(divide='ignore'):
        # ignore infinite value warning & return inf anyway.
        return 2 * O * np.log(O/E) + 2 * (N - O) * np.log((N - O)/(N - E))

def sum_sse(O, E):
    return sum( (O - E)**2 )

def AIC(model=None, LL=None, k=None,):
    if model is not None:
        O = np.concatenate([model.p_signal, model.p_noise])
        E = np.concatenate([model.expected_p_signal, model.expected_p_noise])
        with np.errstate(divide='ignore'):
            logdiff = np.log(np.abs(O - E))
        logdiff[logdiff == -np.inf] = 0
        LL = -(logdiff).sum()
        k = model.k
    return 2 * k - 2 * np.log(LL)

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
        self.non_c_idx = len(self.parameters) # The index number of x0 at which the criterian begin
        self.non_c_labels = list(self.parameters.keys())
        if self.has_criteria:
            self.parameters = self.parameters | {f"c{i+1}": 0 for i in range(len(self.signal)-1)}
            self.parameter_boundaries += [(None, None) for _ in range(len(self.signal)-1)]
        self.parameter_labels = list(self.parameters.keys())
        self.k = len(self.parameter_labels)
        self.x0 = list(self.parameters.values())
    
    def define_model_inputs(self, x0, ignore_criteria=False):
        # Makes dict to unpack as kwargs to the fit function.
        model_input = {k: v for k, v in zip(self.parameter_labels, x0[:self.non_c_idx])}
        if self.has_criteria and not ignore_criteria:
            model_input['criteria'] = x0[self.non_c_idx:]
        return model_input
    
    def get_non_c(self):
        return {k: v for k, v in self.fitted_parameters.items() if k in self.non_c_labels}

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
        
        # Define the model inputs
        model_input = self.define_model_inputs(x0)
        
        # Compute the expected probabilities using the model function
        expected_p_noise, expected_p_signal = self.compute_expected(**model_input)
        
        # Compute the expected counts
        expected_signal = expected_p_signal * self.n_signal
        expected_noise = expected_p_noise * self.n_noise

        # Compute the fit statistic given observed and model-expected data
        if method == 'log-likelihood':
            # Fit using same approach as in spreadsheet
            ll_signal = loglik(O=observed_signal, E=expected_signal, N=self.n_signal)
            ll_noise = loglik(O=observed_noise, E=expected_noise, N=self.n_noise)
            return sum(ll_signal + ll_noise)
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
        # Run the fit function
        self.optimisation_output = minimize(
            fun=self.objective,
            x0=self.x0,
            args=(method),
            bounds=self.parameter_boundaries,
            tol=1e-6
        )
        self.x0 = self.optimisation_output.x
        self.fitted_parameters = {k: v for k, v in zip(self.parameter_labels, self.x0)}
        # Define the model inputs
        self.fitted_model_input = self.define_model_inputs(list(self.fitted_parameters.values()))
        # Compute the expected probabilities using the model function, useful for AIC for example
        self.expected_p_noise, self.expected_p_signal = self.compute_expected(**self.fitted_model_input)
        
        self.results = {
            'fun': self.optimisation_output.fun,
            'success': self.optimisation_output.success,
            'AIC': AIC(self)
        }
        return self.results, self.fitted_parameters
    
    def euclidean_misfit(self, ox, oy, ex, ey):
        pass

class HighThreshold(BaseModel):
    __modelname__ = 'High Threshold'
    has_criteria = False

    def __init__(self, signal, noise):
        self.parameters = {'R': 0.999}
        self.parameter_boundaries = [(0, 1)]
        super().__init__(signal, noise)
    
    def compute_expected(self, R, full=False):
        if full:
            model_noise = np.array([0, 1])
        else:
            model_noise = self.p_noise
        model_signal = (1 - R) * model_noise + R
        return model_noise, model_signal

class SignalDetection(BaseModel):
    __modelname__ = 'Equal Variance Signal Detection'
    has_criteria = True

    def __init__(self, signal, noise, equal_variance=True):
        # .parameters refers to the INPUT parameters defined for the model. Not the fitted_parameters.
        self.parameters = {'d': 0} # Will be updated with all criterion params on super() call
        self.parameter_boundaries = [(None, None)] # also updated on super() to include criteria
        
        if not equal_variance:
            self.__modelname__ = self.__modelname__.replace('Equal', 'Unequal')
            # Add additional parameter for the signal variance, along with boundary
            # As per the docs (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html)...
            #    ...we can actually just include this as a parameter and fix the bounds equal to 1...
            #    ... but the issue is that it would appear as an additional parameter and inflate the d.f.
            self.parameters['scale'] = 1
            self.parameter_boundaries.append((1, 100))
        
        super().__init__(signal, noise)
        
    # SDT function to get f_exp
    def compute_expected(self, d=None, scale=1, criteria=None, full=False):
        # `full` may be unnecessary.
        if criteria is None or full:
            criteria = np.arange(-5, 5, 0.01)
        
        model_signal = stats.norm.cdf(d / 2 - np.array(criteria), scale=scale)
        model_noise = stats.norm.cdf(-d / 2 - np.array(criteria), scale=1)
        return model_noise, model_signal

if __name__ == '__main__':
    pass