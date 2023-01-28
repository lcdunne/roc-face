# Signal Detection

Compute basic signal detection measures & fit theoretical recognition memory models to data. Measures include $d^\prime$ and $c$ (and others; see example 1). Supported models are high-threshold, equal- and unqeual-variance signal detection, and dual-process signal detection (example 2 shows the equal- and unequal-variance models).

## Setting Up

The dependencies can be found in `requirements.txt` and this code needs them to run. I recommend using a virtual environment to isolate the dependencies from your base python installation (for more info [please see this](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)).

To install them, `cd` to the directory with this code in it, and then run `$ pip install -r requirements.txt`.

## Example Usage

### Example 1: Basic signal detection theory measures:

```python
>>> import measures as sdt

>>> sdt.d_prime(0.75, 0.21)
1.480910997214322

>>> sdt.c_bias(0.75, 0.21)
0.06596574841107933

>>> sdt.a_prime(0.75, 0.21)
0.850886075949367

>>> sdt.beta(0.75, 0.21)
0.9761153407498215

>>> sdt.beta_doubleprime(0.75, 0.21)
0.06112054329371819

>>> sdt.beta_doubleprime(0.75, 0.21, donaldson=True)
0.1126760563380282
```

### Example 2: Receiver operating characteristic (ROC) modelling:

```python
>>> from models import SignalDetection

# Strongest "signal" <---> Strongest "noise"
# All responses to signal-present trials
>>> signal = [505,248,226,172,144,93]

# All responses to signal-absent (i.e. noise) trials
>>> noise = [115,185,304,523,551,397]

# Create an equal-variance signal detection model
>>> evsd = SignalDetection(signal, noise)

# Create an unequal-variance signal detection model
>>> uvsd = SignalDetection(signal, noise, equal_variance=False)
```

Once a model has been instantiated, we can view it in ROC space and see the AUC (note that the AUC corresponds to the observed data, rather than to any specific model, and is therefore identical for the `evsd` and `uvsd`). For example:

```python
# Utility plot function
# not required - can just use standard matplotlib
>>> import matplotlib.pyplot as plt
>>> from utils import plot_roc

# Plot the original datapoints
>>> plot_roc(signal, noise); plt.show()

>>> print(evsd.auc)
0.7439343243677308
```
<img src="https://github.com/lcdunne/signal-detection/raw/develop/example/simple_ROC.png" alt="" width="620">

We can fit the two models (`evsd` and `uvsd`) as follows:

```python
# Fit the models using the G^2 fit function
>>> evsd.fit()
{
    'd': 1.020144525302289,
    'criteria': array([ 0.94589698,  0.47680529,  0.01204417, -0.56213984, -1.28720496])
}

>>> uvsd.fit()
{
    'd': 1.1924959229611845,
    'scale': 1.3447947556571425,
    'criteria': array([ 1.03065803,  0.45976385, -0.06872691, -0.70004907, -1.46072399])
}

# Check the results
>>> print(evsd.results)
{
    'model': 'Equal Variance Signal Detection',
    'fit-success': True,
    'fit-method': 'G',
    'statistic': 81.23108616253239,
    'log_likelihood': -8721.468296830686,
    'AIC': 17454.936593661372,
    'BIC': 17491.835936927786,
    'SSE': 0.0062687683639536295
}

>>> print(uvsd.results)
{
    'model': 'Unequal Variance Signal Detection',
    'fit-success': True,
    'fit-method': 'G',
    'statistic': 3.9020247244641175,
    'log_likelihood': -8682.803766111652,
    'AIC': 17379.607532223305,
    'BIC': 17422.65676603412,
    'SSE': 0.0009699101952450413
}
```

They can also be compared statistically, since the $G^2$ statistic is approximately $\chi^2$ distributed. To accomplish this, we just need to (1) compute the *difference* in $G^2$ and (2) compute the *difference* in the degrees of freedom between the two models. This is possible using either the G-test or the $χ^2$ test statistics. These values can then be used to obtain a $p$ value:

```python
>>> from scipy import stats

# Get the difference in G-test values
>>> gdiff = evsd.results['statistic'] - uvsd.results['statistic']

# Get the difference in the degrees of freedom for each model
>>> dofdiff = evsd.dof - uvsd.dof

# Get a p value
>>> p = stats.chi2.sf(x=gdiff, df=dofdiff)

>>> print(f"G({dofdiff}) = {gdiff}, p = {p}")
G(1) = 77.32906143806828, p = 1.4472085251058265e-18
```

Finally, we can just view the ROC data and the two fitted models, as follows:

```python
>>> from utils import plot_zroc

>>> fig, ax = plt.subplots(1, 2, dpi=150)

>>> ax[0].plot(
        *evsd.compute_expected(evsd.fitted_parameters['d']),
        label=evsd.label
    )

>>> ax[0].plot(
        *uvsd.compute_expected(
            uvsd.fitted_parameters['d'],
            uvsd.fitted_parameters['scale']
        ),
        label=uvsd.label
    )

# Plot the original datapoints
>>> plot_roc(signal, noise, c='k', ax=ax[0])

# Plot z-ROC with second-order polynomial fit to the second subplot axis
>>> plot_zroc(signal, noise, reg=True, poly=2, c='k', ax=ax[1])

>>> ax[0].legend(loc='lower right')

>>> plt.tight_layout()
>>> plt.show()
```
<img src="https://github.com/lcdunne/signal-detection/raw/develop/example/example_EVSD_UVSD.png" alt="" width="620">
