# Signal Detection

Compute basic signal detection measures & fit theoretical models to data. Measures include $d^\prime$ and $c$ (and others; see example 1). Supported models are high-threshold, equal- and unqeual-variance signal detection, and dual-process signal detection (example 2 shows the equal- and unequal-variance models).

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
>>> from utils import plot_roc

# Just plots the original datapoints
>>> plot_roc(evsd.p_signal, evsd.p_noise, c='k')

>>> print(evsd.auc)
0.7439343243677308
```

[!data-ROC](https://raw.githubusercontent.com/lcdunne/signal-detection/main/_simple_ROC.png)

We can fit the two models (`evsd` and `uvsd`) as follows:

```python
# Fit the models using the G^2 fit function
>>> evsd.fit()
{
    'd': 1.0201477503320475,
    'criteria': array([ 0.94589746,  0.47680517,  0.01204214, -0.56213821, -1.28720453])
}

>>> uvsd.fit()
{
    'd': 1.192497247996845,
    'scale': 1.3447964928242155,
    'criteria': array([ 1.03065819,  0.45976337, -0.06872705, -0.70004942, -1.46072412])
}

# Check the results
>>> print(evsd.results)
{
    'model': 'Equal Variance Signal Detection',
    'opt-success': True,
    'log-likelihood': 81.23108620009691,
    'aic': -64.79662553538728,
    'bic': -27.897282268972745,
    'euclidean_fit': 0.10630270614413172
}

>>> print(uvsd.results)
{
    'model': 'Unequal Variance Signal Detection',
    'opt-success': True,
    'log-likelihood': 3.902024727392697,
    'aic': -98.43321886926276,
    'bic': -55.3839850584458,
    'euclidean_fit': 0.02829386287310158
}
```

They can also be compared statistically, since the $G^2$ statistic is approximately $\chi^2$ distributed. To accomplish this, we just need to (1) compute the *difference* in $G^2$ and (2) compute the *difference* in the degrees of freedom between the two models. These values can then be used to obtain a $p$ value:

```python
>>> from scipy import stats

# Get the difference in G^2 values
>>> g2diff = evsd.results['log-likelihood'] - uvsd.results['log-likelihood']

# Get the difference in the degrees of freedom for each model
>>> ddofdiff = evsd.ddof - uvsd.ddof

# Get a p value
>>> p = stats.chi2.sf(x=g2diff, df=ddofdiff)

>>> print(f"G^2 ({ddofdiff}) = {g2diff}, p = {p}")
G^2 (1) = 77.32906147270421, p = 1.4472084997268971e-18
```

Finally, we can just view the ROC data and the two fitted models, as follows:

```python
>>> import matplotlib.pyplot as plt

>>> fig, ax = plt.subplots(dpi=150)

>>> ax.plot(
        *evsd.compute_expected(evsd.fitted_parameters['d']),
        label=evsd.label
    )

>>> ax.plot(
        *uvsd.compute_expected(
            uvsd.fitted_parameters['d'],
            uvsd.fitted_parameters['scale']
        ),
        label=uvsd.label
    )

# Just plots the original datapoints
>>> plot_roc(evsd.p_signal, evsd.p_noise, ax=ax, c='k', zorder=999)

>>> ax.legend(loc='lower right')
>>> plt.show()
```

[!model-ROCs](https://raw.githubusercontent.com/lcdunne/signal-detection/main/_example_evsd-uvsd.png)

## Useful links:

[Intro to SDT](https://www.birmingham.ac.uk/Documents/college-les/psych/vision-laboratory/sdtintro.pdf)
