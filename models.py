import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Union, Optional
from base import _BaseModel
from utils import plot_roc, array_like

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
    TODO

    """
    __modelname__ = 'High Threshold'
    _has_criteria = False

    def __init__(self, signal, noise):
        self._named_parameters = {'R': {'initial': 0.99, 'bounds': (0, 1)}}
        self._n_named_parameters = len(self._named_parameters) + 1 # Required because `g` (guess) parameter is implicit
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
            model_noise = self.obs_noise.roc

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
    TODO

    """
    __modelname__ = 'Equal Variance Signal Detection'
    _has_criteria = True

    def __init__(self, signal, noise, equal_variance=True):
        self._named_parameters = {
            'd': {'initial': 1, 'bounds': (None, None)}, # d may need to start above the likely value for convergence with some fit statistics
            # 'scale': {'initial': 1, 'bounds': (1, 1 if equal_variance else None)},
        }
        
        if not equal_variance:
            self.__modelname__ = self.__modelname__.replace('Equal', 'Unequal')
            self._named_parameters['scale'] = {'initial': 1, 'bounds': (1, None)}
        
        self.label = ''.join([i[0] for i in self.__modelname__.split()])
        super().__init__(signal, noise)
    
    @property
    def scale(self):
        """float: the standard deviation (scale) of the signal distribution."""
        return self.fitted_parameters.get('scale', 1.0)

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
    TODO
    
    """
    __modelname__ = 'Dual Process Signal Detection'
    _has_criteria = True

    def __init__(self, signal, noise):
        self._named_parameters = {
            'd': {'initial': 1, 'bounds': (None, None)},
            'R': {'initial': 0.999, 'bounds': (0, 1)},
        }
        
        self.label = ''.join([i[0] for i in self.__modelname__.split()])
        super().__init__(signal, noise)
    
    @property
    def familiarity(self):
        """float: Estimate of familiarity."""
        if not hasattr(self, 'fitted_parameters'):
            return None
        d = self.fitted_parameters.get('d')
        c_x = self.fitted_parameters['criteria'][self.signal_boundary]
        return stats.norm.cdf( d / 2 - c_x )
    
    @property
    def recollection(self):
        """float: Estimate of recollection."""
        if not hasattr(self, 'fitted_parameters'):
            return None        
        return self.fitted_parameters.get('R')

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

        model_noise = stats.norm.cdf(-d / 2 - criteria)
        model_signal = R + (1 - R) * stats.norm.cdf(d / 2 - criteria)

        return model_noise, model_signal
    

if __name__ == '__main__':
    
    signal = [505,248,226,172,144,93]
    noise = [115,185,304,523,551,397]

    fit_method = 'G'
    alt = True # When false, matches ROC toolbox but gives poor fit for HT model.
    
    ht = HighThreshold(signal, noise)
    ht.fit(fit_method, alt=alt)
    print(ht.results)
    
    evsd = SignalDetection(signal, noise, equal_variance=True)
    evsd.fit(fit_method, alt=alt)
    print(evsd.results)
    
    uvsd = SignalDetection(signal, noise, equal_variance=False)
    uvsd.fit(fit_method, alt=alt)
    print(uvsd.results)

    dpsd = DualProcess(signal, noise)
    dpsd.fit(fit_method, alt=alt)
    print(dpsd.results)
    
    # Plot ROC curves
    fig, ax = plt.subplots(dpi=150)
    plot_roc(evsd.obs_signal.roc, evsd.obs_noise.roc, ax=ax)
    ax.plot(*ht.compute_expected(**ht.fitted_parameters), label=ht.label)
    ax.plot(*evsd.compute_expected(**evsd.fitted_parameters), label=evsd.label)
    ax.plot(*uvsd.compute_expected(**uvsd.fitted_parameters), label=uvsd.label)
    ax.plot(*dpsd.compute_expected(**dpsd.fitted_parameters), label=dpsd.label)
    ax.legend(loc='lower right')
    plt.show()
    
<<<<<<< HEAD
    # Plot convergence for each model
    fig, ax = plt.subplots(2,2, dpi=150)
    for axis, model in zip(ax.flatten(), [ht, evsd, uvsd, dpsd]):
        axis.plot(model.convergence)
        axis.set(xlabel='iteration', ylabel=fit_method, title=model.label)    
    plt.tight_layout()
    plt.show()
=======
    fig, ax = plt.subplots(1, 3, figsize=(9,4), dpi=100, sharey=True)
    ax[0].bar(x=np.arange(1,11), height=ht.squared_errors)
    ax[1].bar(x=np.arange(1,11), height=evsd.squared_errors)
    ax[2].bar(x=np.arange(1,11), height=uvsd.squared_errors)
    ax[0].set(ylabel='Log Euclidean Fit', xlabel='criterion', yscale='log', title='High Threshold')
    ax[1].set(title='Equal Variance', yscale='log', xlabel='criterion',)
    ax[2].set(title='Unequal Variance', yscale='log', xlabel='criterion',)
    plt.show()
>>>>>>> cb4678bdd06242228e1891c1e364e976f0819da1
