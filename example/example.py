import json
import logging
import os
from pprint import pprint
import matplotlib.pyplot as plt
from signal_detection import utils
from signal_detection.models import HighThreshold, SignalDetection, DualProcess

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_example_data(dataset_name):
    """Load one of the example datasets.
    
    For the purpose of testing and giving examples, some example responses 
    from published articles are used.
    
    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load. Must be one of the following:
            'OC68': Ogilvie & Creelman (1968)
            'StanTod99': Stanislaw & Todorov (1999)
            'Dunn2011': Dunn (2011)
            'Koen': The ROC toolbox (Tutorial 1)
            'Dunne': A subset of data from the author's unpublished work.
        
    Returns
    -------
    tuple
        A tuple of two lists. The first element is the signal and the second is
        the noise array. The data are response frequencies.
    """
    with open('example/example_data.json', 'r') as f:
        data = json.load(f)
    
    dataset = data.get(dataset_name.upper())

    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets are: {[d for d in data]}.")
    
    return dataset['signal'], dataset['noise']

signal, noise = load_example_data('koen')
# --------------------------------------------------------------------------- #
# Specify data
# signal = []
# noise = []

# Define the models
ht = HighThreshold(signal, noise)
evsd = SignalDetection(signal, noise)
uvsd = SignalDetection(signal, noise, equal_variance=False)
dpsd = DualProcess(signal, noise)

# Fit the models
ht.fit(alt=False)
evsd.fit(alt=False)
uvsd.fit(alt=False)
dpsd.fit('SSE', alt=False)

# Show results
pprint(ht.results)
pprint(evsd.results)
pprint(uvsd.results)
pprint(dpsd.results)

pprint(dpsd.fitted_named_parameters)
pprint({'Recollection': dpsd.recollection, 'Familiarity': dpsd.familiarity})

# Plot a model
fig, ax = plt.subplots(1, 2, dpi=150)

utils.plot_roc(ht.obs_signal.roc, ht.obs_noise.roc, c='k', ax=ax[0])
ax[0].plot(*ht.compute_expected(**ht.fitted_parameters, full=True), label='HT')
ax[0].plot(*evsd.compute_expected(**evsd.fitted_named_parameters), label='EVSD')
ax[0].plot(*uvsd.compute_expected(**uvsd.fitted_named_parameters), label='UVSD')
ax[0].plot(*dpsd.compute_expected(**dpsd.fitted_named_parameters), label='DPSD')

ax[0].legend(loc='lower right')

utils.plot_zroc(ht.obs_signal.roc, ht.obs_noise.roc, poly=2, c='k', ax=ax[1])
plt.tight_layout()
plt.show()