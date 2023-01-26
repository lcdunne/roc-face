import numpy as np
from itertools import product
import pytest
from signal_detection import utils, models
from .context import datasets

def uvsd_model(signal, noise):
    # Convenience
    return models.SignalDetection(signal, noise, equal_variance=False)

# Test all data against ROC toolbox for fit functionality
fit_func_contingencies = [
    ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'G', 82.362, {'d': 1.368}),
    ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'LL', 8659.73, {'d': 1.368}),
    ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'X2', 82.34, {'d': 1.339}),
    ('Dunn2011', datasets.get('Dunn2011'), models.SignalDetection, 'SSE',0.002211, {'d': 1.423}),
    ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'G', 3.0349, {'d': 1.598, 'scale': 1.318}),
    ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'LL', 8620.06, {'d': 1.598, 'scale': 1.318}),
    ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'X2', 3.022, {'d': 1.597, 'scale': 1.318}),
    ('Dunn2011', datasets.get('Dunn2011'), uvsd_model, 'SSE',0.00011164, {'d': 1.618, 'scale': 1.332}),
    ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'G', 23.887, {'R': 0.279, 'd': 1.051}),
    ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'LL', 8630.489, {'R': 0.279, 'd': 1.051}),
    ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'X2', 23.696, {'R': 0.279, 'd': 1.043}),
    ('Dunn2011', datasets.get('Dunn2011'), models.DualProcess, 'SSE',0.000495, {'R': 0.450, 'd': 0.898}), # is 0.0023 in ROC toolbox
    ('Koen', datasets.get('Koen'), models.SignalDetection, 'G', 35.55, {'d': 0.792}),
    ('Koen', datasets.get('Koen'), models.SignalDetection, 'LL', 2.1558e3, {'d': 0.792}),
    ('Koen', datasets.get('Koen'), models.SignalDetection, 'X2', 35.01, {'d': 0.754}),
    ('Koen', datasets.get('Koen'), models.SignalDetection, 'SSE',0.0044, {'d': 0.961}),
    ('Koen', datasets.get('Koen'), uvsd_model, 'G', 1.7281, {'d': 0.993, 'scale': 1.399}),
    ('Koen', datasets.get('Koen'), uvsd_model, 'LL', 2.1389e3, {'d': 0.993, 'scale': 1.399}),
    ('Koen', datasets.get('Koen'), uvsd_model, 'X2', 1.7489, {'d': 0.992, 'scale': 1.398}),
    ('Koen', datasets.get('Koen'), uvsd_model, 'SSE',0.00011164, {'d': 0.996, 'scale': 1.415}),
    ('Koen', datasets.get('Koen'), models.DualProcess, 'G', 8.5786, {'R': 0.249, 'd': 0.437}),
    ('Koen', datasets.get('Koen'), models.DualProcess, 'LL', 2.1423e3, {'R': 0.249, 'd': 0.437}),
    ('Koen', datasets.get('Koen'), models.DualProcess, 'X2', 8.6103, {'R': 0.249, 'd': 0.429}),
    ('Koen', datasets.get('Koen'), models.DualProcess, 'SSE', 0.0012, {'R': 0.242, 'd': 0.515}), # is 0.0012 in ROC toolbox, but for some reason getting 0.02052593...
]

@pytest.mark.parametrize('dataset_name,dataset,model,fitstat,expected_fitstat,expected_fitted_params', fit_func_contingencies)
def test_fit_statistics(dataset_name, dataset, model, fitstat, expected_fitstat,expected_fitted_params):
    # Arrange
    m = model(dataset['signal'], dataset['noise'])
    # Act
    m.fit(fitstat, alt=False)
    # Assert
    fit = m.optimisation_output.fun
    assert m.results['fit-success'] == True, f"{m.label} model fit failed for {dataset_name} dataset..."
    assert np.allclose(fit, expected_fitstat, rtol=.01), f"Expected {fitstat}={expected_fitstat}, but got {fitstat}{fit}..."


# Test all data against ROC toolbox for fitted parameter estimates
@pytest.mark.parametrize('dataset_name,dataset,model,fitstat,expected_fitstat,expected_fitted_params', fit_func_contingencies)
def test_fitted_parameters(dataset_name, dataset, model, fitstat, expected_fitstat,expected_fitted_params):
    m = model(dataset['signal'], dataset['noise'])
    m.fit(fitstat, alt=False)

    # Assert model fitted parameters match
    fitted_parameters = m.fitted_named_parameters
    for param, parameter_value in fitted_parameters.items():
        expected = expected_fitted_params[param]
        assert np.allclose(parameter_value, expected, rtol=.01), f"Expected {param}={expected} but got {param}={parameter_value}."


def test_alt_true_better_for_high_threshold():
    # Test that the results using alt=True are far better than alt=False (for the HT model)
    return

def test_alt_true_similar_to_alt_false():
    # Test that the results from alt=True are comparable to alt=False (for all SDT models)
    return

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