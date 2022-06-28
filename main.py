import logging
import numpy as np
import pandas as pd # for comparisons
from scipy import stats
from scipy.optimize import minimize
import legacy_chi2 as legacy_stats

np.random.seed(42)

class BaseModel:
    def __init__(self, signal, noise):
        self.signal = signal
        self.noise = noise
        self._set_criterion_parameters()

    def _set_criterion_parameters(self):
        if hasattr(self, 'criteria') and self.criteria:
            self.criteria = np.random.normal(0, 1, len(self.signal))
        else:
            self.criteria = False
        


class Model(BaseModel):
    def __init__(self, signal, noise):
        self.parameters = dict(d=0, R=0.1)
        self.constraints = dict(d=(None, None), r=(0, 1))
        self.criteria = True
        super().__init__(signal, noise)

#%% utils
def accumulate(arr):
    return np.cumsum(arr)

def corrected_proportion(a, x, i=0):
    """Convert the response rates to probabilities.

    Applies a correction to the rates such that if the next value is the
    same, i.e. a subject made zero responses for the next category, then
    the probability is still slightly larger by a very small amount.

    Applies a correction to the rates such that all except for the last
    value are < 1. This prevents invalid values for later calculations
    (Stanislaw & Todorov, 1999).

    Parameters
    ----------
    a : [type]
        [description]
    x : [type]
        [description]
    i : int, optional
        [description], by default 0

    Returns
    -------
    [type]
        [description]
    """
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

def chitest(obs, exp):
    return sum(((np.array(obs)-np.array(exp))**2) / np.array(exp))


def gtest(N, n_c, obs_c, expct_c):
    # print(f"N: {N}\tn_c: {n_c}\tobs_c: {obs_c}\texpct_c: {expct_c}")
    """Vectorised G\ :sup:`2` test.

    Parameters
    ----------
    N : int
        The sum of response counts across all scale levels for this class.
    n_c : [type]
        [description]
    obs_c : [type]
        B2:B7 (if hits); C2:C7 (if FAs)
    expct_c : [type]
        ~E22:~E26 (if hits); ~E37:~E31 (if FAs)

    Returns
    -------
    [type]
        [description]
    """
    # logger.info(f"N = {N}\tn_c = {n_c}\tobs_c = {obs_c}")
    with np.errstate(divide='ignore'):
        # Ignore the infinite value warning & return inf anyway.
        g = 2 * n_c * np.log(obs_c / expct_c) + \
            2 * (N - n_c) * np.log((1 - obs_c) / (1 - expct_c))
    return g


#%% Base

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

class BaseModel:
    """Base class for all models.
    
    Contains functionality common to all models but is not intended to be accessed directly.
    """
    __modelname__ = 'Unnamed'

    def __init__(self, signal, noise):
        self.signal = signal
        self.noise = noise

        if not arrays_equal_length(signal, noise):
            raise ValueError(
                "Expected signal and noise to have the same length. "
                f"Signal was length {len(signal)}; "
                f"noise was length {len(noise)}."                
            )

        self.p_signal = compute_proportions(self.signal)
        self.p_noise = compute_proportions(self.noise)
        self._non_c_labels = list(self.parameters.keys())

        if self.use_criteria:
            np.random.seed(RANDOM_SEED)
            self.criteria = np.random.rand(len(self.p_signal))

        self.opt = False

    def __repr__(self):
        return f'<{self.__modelname__}>'

    def _add_criteria_values(self):
        """Expands the parameters class attribute to include criterion
        values."""
        criteria = {}
        for i, c in enumerate(np.random.rand(len(self.p_signal))):
            criteria[f'c{i}'] = {'value': c, 'bounds': (None, None)}
        self.parameters = {**self.parameters, **criteria}

    def _prepare_solver_input(self):
        """Extracts the parameters and their boundaries"""
        x0 = [v['value'] for v in self.parameters.values()]
        x0_bounds = [v['bounds'] for v in self.parameters.values()]
        if self.use_criteria:
            x0 += list(self.criteria)
            x0_bounds += [(None, None)] * len(self.p_signal)
        return x0, x0_bounds

    def _map_x0_to_dict(self, x0):
        n_params = len(self.parameters)
        # Zip the inputted param keys with x0 vals @ those indexes.
        to_fit = dict(zip(self.parameters.keys(), x0[:n_params]))
        if self.use_criteria:
            to_fit['c'] = x0[n_params:]
        return to_fit

    def _sum_g2(self, x0):
        """
        Input x0 is a flat array containing all parameters to update.
        
        Computes expected and returns them as expected signal & noise arrays.
        Rescales the data to count for compatibility with scipy's fit function.
        Computes fit statistics for signal and noise at each response level.
        Returns the sum to use as the objective function.
        
        This might need generalising
        Example:
            
            gsquares, _ = stats.power_divergence(
                f_obs=[
                    [504.8029692, 752.7909747, 978.7948229, 1150.837534, 1294.900414],
                    [883.1970308, 635.2090253, 409.2051771, 237.1624660, 93.09958644]
                ],
                f_exp=[
                    [518.1906456, 750.1026269, 957.0487077, 1155.456118, 1300.471984],
                    [869.8093544, 637.8973731, 430.9512923, 232.5438822, 87.52801564]
                ],
                lambda_='log-likelihood',
            )
            print(gsquares)
            
            >>> [0.55390143 0.02096917 1.60667828 0.10961499 0.37126655]
        """
        # Compute the expected values for the current iteration
        # We can unpack the results of _map_x0_to_dict in order to get it.
        self.expected_noise, self.expected_signal = self._compute_expected(
            **self._map_x0_to_dict(x0)
        )
        
        N_signal, N_noise = sum(self.signal), sum(self.noise)
        
        observed_signal = self.p_signal * N_signal
        observed_noise = self.p_noise * N_noise
        expected_signal = self.expected_signal * N_signal
        expected_noise = self.expected_noise * N_noise
        
        # Make use of legacy scipy function.
        # Current implementation forces use of equally-summed f_obs and f_exp which is unreasonable (see https://github.com/scipy/scipy/issues/14298).
        # An alternative approach would be to manually scale the inputs to match, making this compatible with current scipy.
        signal_gval, _ = legacy_stats.power_divergence(
            f_obs=[observed_signal, N_signal - observed_signal],
            f_exp=[expected_signal, N_signal - expected_signal],
            lambda_='log-likelihood'
        )
        
        noise_gval, _ = legacy_stats.power_divergence(
            f_obs=[observed_noise, N_noise - observed_noise],
            f_exp=[expected_noise, N_noise - expected_noise],
            lambda_='log-likelihood'
        )
        
        return sum(signal_gval) + sum(noise_gval)

    def _sum_chi2(self, x0):
        """
        Input x0 is a flat array containing all parameters to update.
        
        Computes expected and returns them as expected signal & noise arrays.
        Rescales the data to count for compatibility with scipy's fit function.
        Computes fit statistics for signal and noise at each response level.
        Returns the sum to use as the objective function.
        
        This might need generalising
        Example:
            
            gsquares, _ = stats.power_divergence(
                f_obs=[
                    [504.8029692, 752.7909747, 978.7948229, 1150.837534, 1294.900414],
                    [883.1970308, 635.2090253, 409.2051771, 237.1624660, 93.09958644]
                ],
                f_exp=[
                    [518.1906456, 750.1026269, 957.0487077, 1155.456118, 1300.471984],
                    [869.8093544, 637.8973731, 430.9512923, 232.5438822, 87.52801564]
                ],
                lambda_='pearson',
            )
            print(gsquares)
            
            >>> [0.55390143 0.02096917 1.60667828 0.10961499 0.37126655]
        """
        # Compute the expected values for the current iteration
        # We can unpack the results of _map_x0_to_dict in order to get it.
        self.expected_noise, self.expected_signal = self._compute_expected(
            **self._map_x0_to_dict(x0)
        )
        
        N_signal, N_noise = sum(self.signal), sum(self.noise)
        
        observed_signal = self.p_signal * N_signal
        observed_noise = self.p_noise * N_noise
        expected_signal = self.expected_signal * N_signal
        expected_noise = self.expected_noise * N_noise
        
        # Make use of legacy scipy function.
        # Current implementation forces use of equally-summed f_obs and f_exp which is unreasonable (see https://github.com/scipy/scipy/issues/14298).
        # An alternative approach would be to manually scale the inputs to match, making this compatible with current scipy.
        signal_chival, _ = legacy_stats.power_divergence(
            f_obs=[observed_signal, N_signal - observed_signal],
            f_exp=[expected_signal, N_signal - expected_signal],
            lambda_='pearson'
        )
        
        noise_chival, _ = legacy_stats.power_divergence(
            f_obs=[observed_noise, N_noise - observed_noise],
            f_exp=[expected_noise, N_noise - expected_noise],
            lambda_='pearson'
        )
        
        return sum(signal_chival) + sum(noise_chival)
    
    def _sum_sse(self, x0):
        """
        Input x0 is a flat array containing all parameters to update.
        
        Computes expected and returns them as expected signal & noise arrays.
        Computes fit statistics for signal and noise at each response level.
        Returns the sum to use as the objective function.
        
        This might need generalising
        Example:
            
        """
        # Compute the expected values for the current iteration
        # We can unpack the results of _map_x0_to_dict in order to get it.
        self.expected_noise, self.expected_signal = self._compute_expected(
            **self._map_x0_to_dict(x0)
        )
        
        signal_sse = sum((self.p_signal - self.expected_signal)**2)
        noise_sse = sum((self.p_noise - self.expected_noise)**2)
        
        return signal_sse + noise_sse

    def fit(self, method='g'):
        """Fit the model using the chosen objective function.

        Parameters
        ----------
        method : str, optional
            Label denoting the objective function to minimize. Can be `'g'`
            or `'chi'`, by default 'g'.

        Raises
        ------
        ValueError
            If the requested objective function is not recognised.
        
        Notes
        -----
        Arranges all model parameters so that they can be input to the
        ``scipy.optimize.minimize`` function. This will adjust all input
        parameters, compute the expected values using ``_compute_expected``,
        and then compute the objective function (`G`\ :sup:`2` or χ\ :sup:`2`). The
        process is repeated until a solution is obtained. See `SciPy's minimization
        function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ for more information.
        """
        x0, x0_bounds = self._prepare_solver_input()
        
        if method == 'g':
            fun = self._sum_g2
        elif method == 'chi':
            fun = self._sum_chi2
        elif method == 'sse':
            fun = self._sum_sse
        else:
            raise ValueError(f"Method '{method}' not valid. Try one 'g' or 'chi'.")

        self.result = minimize(
            x0=x0, bounds=x0_bounds,
            fun=fun, tol=1e-6
        )

        # Loop on the parameter labels & store the fitted
        for i, param in enumerate(self.parameters.keys()):
            self.parameters[param]['fitted'] = self.result.x[i]

        # Map the fitted x0 to a dictionary
        self.fitted_parameters = self._map_x0_to_dict(self.result.x)
        self.opt = self.result.success
        if not self.opt:
            # Raise a warning...?
            logger.warning("Solver failed to converge!")
        # Re-fit
        self._compute_expected(**self.fitted_parameters)

#%% Models
class SignalDetection(BaseModel):
    __modelname__ = 'Signal Detection'

    use_criteria = True

    def __init__(self, signal, noise, equal_variance=True):
        self.equal_variance = equal_variance
        self.parameters = dict(d=dict(value=1, bounds=(None, None)))

        if not self.equal_variance:
            self.parameters.update(old_variance=dict(value=1, bounds=(0, None)))
            _pref = 'Unequal Variance '
        else:
            _pref = 'Equal Variance '
        self.__modelname__ = _pref + self.__modelname__
        super().__init__(signal, noise)

    def _compute_expected(self, d=None, c=None, old_variance=None):
        if self.opt:
            # If optimized, make the curve extend from x=0 through x=1
            c = np.linspace(-5, 5, 501)
            if self.equal_variance:
                self.noise_ratio = 1
            else:
                self.noise_ratio = 1 / self.fitted_parameters['old_variance']

        self.model_signal = stats.norm.cdf(d / 2 - c, scale=old_variance or 1)
        self.model_noise = stats.norm.cdf(-d / 2 - c, scale=1)
        return self.model_noise, self.model_signal

def compare(*models):
    """Compare different models for their fits."""
    y = {'model': [], 'y': []}
    for model in models:
        y['model'].append(model.__modelname__)
        y['y'].append(model.result.fun)
    return pd.DataFrame(y).sort_values(by='y', ascending=False)

import matplotlib.pyplot as plt

def plot_roc(model, data=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.axis('square')
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.plot([0,1], [0,1], c='k', lw=1, ls='dashed')
    if data:
        ax.scatter(model.p_noise, model.p_signal)
    ax.plot(model.expected_noise, model.expected_signal, label=model.__modelname__)
    ax.legend()
    return ax

#%% Testing
# High confidence >...> High confidence "noise"
signal = [505, 248, 226, 172, 144, 93]
# High confidence >...> High confidence "noise"
noise = [115, 185, 304, 523, 551, 397]

evsd = SignalDetection(signal, noise)
uvsd = SignalDetection(signal, noise, False)

evsd.fit('g')
uvsd.fit('g')

print(
    compare(
        evsd,
        uvsd,
    )
)
fig, ax = plt.subplots(dpi=100)
plot_roc(evsd, data=True, ax=ax)
plot_roc(uvsd, ax=ax)
plt.show()
