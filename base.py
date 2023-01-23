import numpy as np
from scipy import stats
from scipy.optimize import minimize
from utils import *

class ResponseData:
    def __init__(self, freqs=None, props_acc=None, n=None, corrected=True):
        if freqs is not None:
            # Create derivatives based on the observed frequencies
            self.freqs = np.array(freqs)
            self.n = sum(self.freqs)
            self.corrected = corrected
            self.props = self.freqs / self.n 
            self.freqs_acc = accumulate(self.freqs)
            self.props_acc = compute_proportions(self.freqs, truncate=False, corrected=self.corrected)

        elif (props_acc is not None) and (n is not None):
            # Create all derived vars from the accumulated proportions. Useful 
            # for deriving expected frequencies based on model predicted 
            # accumulated propotions.
            # Note that this is not the reverse of defining ResponseData with 
            # freqs, unless the `corrected` argument to `compute_proportions` 
            # is set to True.
            self.n = n
            self.corrected = False
            self.props_acc = np.append(props_acc, 1)
            self.freqs_acc = self.props_acc * self.n
            self.freqs = deaccumulate(self.freqs_acc)
            self.props = deaccumulate(self.props_acc)
        else:
            raise ValueError("Either `freqs` or both of `props_acc` and `n` are required.")
        self.z = stats.norm.ppf(self.roc)
    
    def __repr__(self):
        return self.table.get_string()
    
    @property
    def table(self):
        return keyval_table(**self.as_dict)
    
    @property
    def as_dict(self):
        return {
            'N': self.n,
            'Freqs.': self.freqs,
            'Freqs. (Accum.)': self.freqs_acc,
            'Props.': self.props,
            'Props. (Accum)': self.props_acc,
            'z-score': self.z,
        }
    
    @property
    def roc(self):
        # Ok to call it ROC? I think so.
        # Test: create ResponseData from props_acc with and without the 1.0. This should behave well.
        return self.props_acc[:-1]

class GenericDataContainer:
    def __init__(self, **kwargs):
        self._inputs = kwargs
        for k, v in self._inputs.items():
            setattr(self, k, v)

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
        self.obs_signal = ResponseData(signal)
        self.obs_noise = ResponseData(noise)
        self.auc = auc(x=self.obs_noise.props, y=self.obs_signal.props)
        
        # Dummy parameters in case no model is specified. This is the fully saturated model (not intended for use).
        if not self._named_parameters:
            _s = {f's{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.obs_signal.props)}
            _n = {f'n{i+1}': {'initial': 0, 'bounds': (None, None)} for i, s in enumerate(self.obs_noise.props)}
            self._named_parameters = _s | _n
        
        # Set the criteria for the model if required
        if self._has_criteria: 
            self.n_criteria = len(self.obs_signal.roc)
            # TODO: Initial starting values for criteria likely need revisiting because can cause model fit errors.
            # Maybe allow the user to optionally pass in a set of "reasonable" starting values.
            # Alternatively, check out the approach taken by Koen in the ROC toolbox.
            self._criteria = {}
            for i, c in enumerate(np.linspace(-0.1, 0.1, self.n_criteria)):
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
        return len(self.obs_signal.roc) + len(self.obs_noise.roc) - self.n_param
    
    # @classmethod ?
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
        return self.obs_noise.roc.copy(), self.obs_signal.roc.copy()


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
        
        self.exp_signal = ResponseData(props_acc=expected_p_signal, n=self.obs_signal.n)
        self.exp_noise = ResponseData(props_acc=expected_p_noise, n=self.obs_noise.n)
        
        if method.upper() == 'SSE':
            sse_signal = squared_errors(self.obs_signal.roc, self.exp_signal.roc).sum()
            sse_noise = squared_errors(self.obs_noise.roc, self.exp_noise.roc).sum()
            return sse_signal + sse_noise
        
        elif method.upper() in ['G', 'X2', 'CHI2', 'CHI']:     
            # lambda_ for power_divergence: 1=chitest, 0=gtest (see SciPy docs)
            lambda_ = int(method.upper() != 'G') # if true, then converted to 1, else 0

            observed = np.array([
                self.obs_signal.freqs_acc[:-1],
                self.obs_signal.n - self.obs_signal.freqs_acc[:-1],
                self.obs_noise.freqs_acc[:-1],
                self.obs_noise.n - self.obs_noise.freqs_acc[:-1]
            ])

            expected = np.array([
                self.exp_signal.freqs_acc[:-1],
                self.obs_signal.n - self.exp_signal.freqs_acc[:-1],
                self.exp_noise.freqs_acc[:-1],
                self.obs_noise.n - self.exp_noise.freqs_acc[:-1]
            ])
            
            gof = stats.power_divergence(
                observed,
                expected,
                lambda_=lambda_,
            )
            # TODO: want to save the p-value?
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
        
        # TODO: After the above, would be nice to have a method to make all stats
        #   like ._make_results()
        
        # Errors
        sse_signal = squared_errors(self.obs_signal.roc, self.exp_signal.roc).sum()
        sse_noise = squared_errors(self.obs_noise.roc, self.exp_noise.roc).sum()
        self.sse = sse_signal + sse_noise
        
        # Compute the -LL, AIC, and BIC
        # Shouldn't this be computed on the accumulated freqs?
        signal_LL = log_likelihood(self.obs_signal.freqs, self.exp_signal.props)
        noise_LL = log_likelihood(self.obs_noise.freqs, self.exp_noise.props)
        self.LL = signal_LL + noise_LL
        self._aic = 2 * self.n_param - 2 * self.LL
        self._bic = self.n_param * np.log(self.obs_signal.n + self.obs_noise.n) - 2 * self.LL
        
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