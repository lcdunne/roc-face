import numpy as np
from itertools import product
import pytest
from .context import utils, models, datasets

# all_models = [
#     (models.HighThreshold, {}),
#     (models.SignalDetection, {'equal_variance': True}),
#     (models.SignalDetection, {'equal_variance': False}),
#     (models.DualProcess, {})
# ]

# Create every combination of dataset & model
# datasets_and_models = []
# for dataset_name, dataset in datasets.items():
#     for model, model_kwargs in all_models:
#         datasets_and_models.append( [dataset_name, dataset, model, model_kwargs] )

# all_contingencies = list(
#     product(
#         datasets.items(), # x5
#         all_models, # x4
#         ['chi', 'g', 'll', 'sse'], #objective (x4)
#         [True, False], # alt (x2)
#     )
# )

# @pytest.mark.parametrize("dataset,model_defs,objective,alt", all_contingencies[:5])
# def test_fit_succeeds(dataset, model_defs, objective, alt):
#     '''Test that the minimisation succeeds for every dataset & every model'''
#     dataset_name=dataset[0]
#     m = model_defs[0](dataset[1]['signal'], dataset[1]['noise'], **model_defs[1])
#     m.fit(objective, alt)
#     assert m.results['fit-success'] == True, f"{m.label} model fit failed for {dataset_name} dataset."


def uvsd_model(signal, noise):
    # Convenience
    return models.SignalDetection(signal, noise, equal_variance=False)

# Test all data against ROC toolbox for fit functionality
fit_func_contingencies = [
    ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'G', 82.362),
    # ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'LL', -8659.73), # bug
    ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'CHI2', 82.34),
    # ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'SSE',0.002211),
    ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'G', 3.0349),
    # ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'LL', -8620.06), # bug
    ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'CHI2', 3.022),
    # ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'SSE',0.00011164),
    ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'G', 23.887),
    # ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'LL', -8630.489), # bug
    ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'CHI2', 23.696),
    # ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'SSE',0.000495), # is 0.0023 in ROC toolbox
    ('Koen', datasets.get('Koen'), models.SignalDetection, 'G', 35.55),
    # ('Koen', datasets.get('Koen'), models.SignalDetection, 'LL', -2.1558e3), # bug
    ('Koen', datasets.get('Koen'), models.SignalDetection, 'CHI2', 35.01),
    # ('Koen', datasets.get('Koen'), models.SignalDetection, 'SSE',0.0044),
    ('Koen', datasets.get('Koen'), uvsd_model, 'G', 1.7281),
    # ('Koen', datasets.get('Koen'), uvsd_model, 'LL', -2.1389e3), # bug
    ('Koen', datasets.get('Koen'), uvsd_model, 'CHI2', 1.7489),
    # ('Koen', datasets.get('Koen'), uvsd_model, 'SSE',0.00011164),
    ('Koen', datasets.get('Koen'), models.DualProcess, 'G', 8.5786),
    # ('Koen', datasets.get('Koen'), models.DualProcess, 'LL', -2.1423e3), # bug
    ('Koen', datasets.get('Koen'), models.DualProcess, 'CHI2', 8.6103),
    # ('Koen', datasets.get('Koen'), models.DualProcess, 'SSE',0.0012), # is 0.0012 in ROC toolbox
]
@pytest.mark.parametrize('dataset_name,dataset,model,fitstat,expected', fit_func_contingencies)
def test_fit_statistics(dataset_name, dataset, model, fitstat, expected):
    # Arrange
    m = model(dataset['signal'], dataset['noise'])
    # Act
    m.fit(fitstat, alt=False)
    # Assert
    fit = m.optimisation_output.fun
    assert m.results['fit-success'] == True, f"{m.label} model fit failed for {dataset_name} dataset..."
    assert np.allclose(fit, expected, atol=0.01), f"Expected {expected}, but got {fit}..."


# Test all data against ROC toolbox for fitted parameter estimates
# fitted_parameter_contingencies = [
#     ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'd', 82.362),
# ]
def test_fitted_parameters():
    # TODO
    return

def test_alt_true_better_for_high_threshold():
    # Test that the results using alt=True are far better than alt=False (for the HT model)
    return

def test_alt_true_similar_to_alt_false():
    # Test that the results from alt=True are comparable to alt=False (for all SDT models)
    return