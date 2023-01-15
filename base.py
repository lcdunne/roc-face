import numpy as np
from scipy import stats
from scipy.optimize import minimize
import legacy_chi2 as legacy_stats
from utils import *

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
        self.z_signal = stats.norm.ppf(self.p_signal)
        self.z_noise = stats.norm.ppf(self.p_noise)
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
        """dict: All parameters and values after fitting."""
        # TODO: Prevent error when calling this before fitting.
        return self._fitted_parameters

    @property
    def fitted_named_parameters(self):
        """dict: Named parameters and values after fitting."""
        return {k: self.fitted_parameters[k] for k in self._named_parameters.keys()}
    
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

    @property
    def aic(self):
        return self._aic

    @property
    def bic(self):
        return self._bic

    @property
    def ddof(self):
        return len(self.p_signal) + len(self.p_noise) - self.n_param
    
    
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
        
        # Compute the expected counts
        self.expected_signal = self.expected_p_signal * self.n_signal
        self.expected_noise = self.expected_p_noise * self.n_noise
        
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
        self._aic = aic(L=L, k=self.n_param)
        self._bic = bic(L=L, k=self.n_param, n=self.n_signal + self.n_noise)
        
        # Compute the overall euclidean fit
        signal_euclidean = euclidean_distance(self.p_signal, self.expected_p_signal)
        noise_euclidean = euclidean_distance(self.p_noise, self.expected_p_noise)
        self.euclidean_fit = signal_euclidean + noise_euclidean
        
        # TODO: Define nice results output
        self.results = {
            'model': self.__modelname__,
            'opt-success': self.optimisation_output.success,
            method: self.optimisation_output.fun,
            'aic': self._aic,
            'bic': self._bic,
            'euclidean_fit': self.euclidean_fit,
        }
        
        return self.fitted_parameters

if __name__ == '__main__':
    signal = [505,248,226,172,144,93]
    noise = [115,185,304,523,551,397]
    
    x = _BaseModel(signal, noise)