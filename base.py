import numpy as np
from scipy import stats
from scipy.optimize import minimize
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
        # Original observed frequencies
        self.signal = np.array(signal)
        self.noise = np.array(noise)
        self.n_signal = sum(self.signal)
        self.n_noise = sum(self.noise)
        
        # Accumulated observed frequencies
        self.acc_signal = accumulate(self.signal)
        self.acc_noise = accumulate(self.noise)
        
        # Accumulated observed frequencies (prob. space)
        self.p_signal = compute_proportions(self.signal)
        self.p_noise = compute_proportions(self.noise)
        
        # Accumulated observed frequencies (z space)
        self.z_signal = stats.norm.ppf(self.p_signal)
        self.z_noise = stats.norm.ppf(self.p_noise)
        
        # Observed AUC
        self.auc = auc(x=np.append(self.p_noise, 1), y=np.append(self.p_signal, 1))
        
        # Dummy parameters in case no model is specified. This is the fully saturated model (not intended for use).
        if not self._named_parameters:
            _s = {f's{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.p_signal)}
            _n = {f'n{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.p_noise)}
            self._named_parameters = _s | _n
        
        # Set the criteria for the model if required
        if self._has_criteria: 
            self.n_criteria = len(self.p_signal)
            criteria = np.linspace(-0.1, 0.1, self.n_criteria)
            
            self._criteria = {}
            for i, c in enumerate(criteria):
                self._criteria[f"c{i}"] = {'initial': c, 'bounds': (None, None)}

            self._parameters = self._named_parameters | self._criteria
        else:
            # Only the case for high threshold model (currently)
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
    def dof(self):
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

    def _objective(self, x0: array_like, method: Optional[str]='G') -> float:
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
        # Define the model inputs as kwargs for the models' compute_expected method.
        # See the specific model's `compute_expected` for more information.
        model_input = self.define_model_inputs(
            labels=self.parameter_labels,
            values=x0,
            n_criteria=self.n_criteria
        )
        
        # Compute the expected probabilities using the model function
        expected_p_noise, expected_p_signal = self.compute_expected(**model_input)
        
        # Compute the expected counts
        # TODO: Make this a function because we want to return this as output
        expected_signal = prop2freq(expected_p_signal, self.n_signal)
        expected_noise = prop2freq(expected_p_noise, self.n_noise)

        
        if method.upper() == 'SSE':
            sse_signal = squared_errors(self.p_signal, expected_p_signal)
            sse_noise = squared_errors(self.p_noise, expected_p_noise)
            return sum(sse_signal + sse_noise)
        
        elif method.upper() in ['G', 'X2', 'CHI2', 'CHI']:     
            # lambda_ for power_divergence: 1=chitest, 0=gtest (see SciPy docs)
            lambda_ = int(method.upper() != 'G') # if true, then converted to 1, else 0

            observed = np.array([
                self.acc_signal[:-1],
                self.n_signal - self.acc_signal[:-1],
                self.acc_noise[:-1],
                self.n_noise - self.acc_noise[:-1]
            ])

            expected = np.array([
                expected_signal,
                self.n_signal - expected_signal,
                expected_noise,
                self.n_noise - expected_noise
            ])
            
            gof = stats.power_divergence(
                observed,
                expected,
                lambda_=lambda_,
            )

            return sum(gof.statistic)

        else:
            raise ValueError(f"Method must be one of SSE, X2, or G, but got {method}.")

    
    def fit(self, method: Optional[str]='G'):
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
        method : str
            The name of the objective function. Currently accepted values are 
            'G', and 'sse'. The default is 'G'.

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

        # Define the model inputs as kwargs for the model's `compute_expected` method        
        self._fitted_parameters = self.define_model_inputs(
            labels=self.parameter_labels,
            values=self.fitted_values,
            n_criteria=self.n_criteria
        )

        # Compute the expected probabilities using the model's function
        self.expected_p_noise, self.expected_p_signal = self.compute_expected(
            **self._fitted_parameters
        )
        
        # Compute the expected counts
        self.expected_signal = self.expected_p_signal * self.n_signal
        self.expected_noise = self.expected_p_noise * self.n_noise
        
        # TODO: After the above, would be nice to have a method to make all stats
        #   like ._make_results()
        
        # Errors
        self.signal_sse = (self.p_signal - self.expected_p_signal) ** 2
        self.noise_sse = (self.p_noise - self.expected_p_noise) ** 2
        self.sse = sum(self.signal_sse) + sum(self.noise_sse)
        
        # # Compute the AIC - TODO: may be incorrect ----------------- #
        # diffs = np.sqrt(self.squared_errors)
        # diffs[diffs == 0] = 1                   # hack 1 (prevent infinite values)
        # L = np.product(diffs)**-1               # hack 2 (make it fit to the AIC function)
        # self._aic = aic(L=L, k=self.n_param)
        # self._bic = bic(L=L, k=self.n_param, n=self.n_signal + self.n_noise)
        # Leaving this here for future reference.
        # ------------------------------------------------------------ #
        
        # Compute the -LL, AIC, and BIC
        signal_LL = log_likelihood(
            np.array(self.signal),
            deaccumulate(np.append(self.expected_p_signal, 1))
        )
        noise_LL = log_likelihood(
            np.array(self.noise),
            deaccumulate(np.append(self.expected_p_noise, 1))
        )
        self.LL = signal_LL + noise_LL
        self._aic = 2 * self.n_param - 2 * self.LL
        self._bic = self.n_param * np.log(self.n_signal + self.n_noise) - 2 * self.LL
        
        # # Compute the overall euclidean fit
        # signal_euclidean = euclidean_distance(self.p_signal, self.expected_p_signal)
        # noise_euclidean = euclidean_distance(self.p_noise, self.expected_p_noise)
        # self.euclidean_fit = signal_euclidean + noise_euclidean
        
        # TODO: Define nice results output
        self.results = {
            'model': self.__modelname__,
            'opt-success': self.optimisation_output.success,
            'method': method,
            'statistic': self.optimisation_output.fun,
            'log_likelihood': self.LL,
            'aic': self._aic,
            'bic': self._bic,
            'SSE': self.sse,
            # 'euclidean_fit': self.euclidean_fit,
        }
        
        return self.fitted_parameters

if __name__ == '__main__':
    signal = [505,248,226,172,144,93]
    noise = [115,185,304,523,551,397]
    
    x = _BaseModel(signal, noise)