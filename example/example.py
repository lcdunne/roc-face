import json
import matplotlib.pyplot as plt
from roc_face import utils
from roc_face.models import HighThreshold, SignalDetection, DualProcess


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
    with open('example_data.json', 'r') as f:
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
ht.fit(cumulative=False)
evsd.fit(cumulative=False)
uvsd.fit(cumulative=False)
dpsd.fit(cumulative=False)

# Show results
print(ht.results)
print(evsd.results)
print(uvsd.results)
print(dpsd.results)

print(dpsd.parameter_estimates)
print({'Recollection': dpsd.recollection, 'Familiarity': dpsd.familiarity})

# Plot a model
fig, ax = plt.subplots(1, 2, dpi=150)

utils.plot_roc(signal, noise, c='k', ax=ax[0])
ax[0].plot(*ht.curve, label='HT')
ax[0].plot(*evsd.curve, label='EVSD')
ax[0].plot(*uvsd.curve, label='UVSD')
ax[0].plot(*dpsd.curve, label='DPSD')

ax[0].legend(loc='lower right')

utils.plot_zroc(signal, noise, poly=2, scatter_kwargs={'c': 'k'}, ax=ax[1])
plt.tight_layout()
plt.show()