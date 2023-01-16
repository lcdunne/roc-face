'''Tests'''
import pytest
import numpy as np
from .context import models

@pytest.fixture
def signal_array():
    return [505,248,226,172,144,93]

@pytest.fixture
def noise_array():
    return [115,185,304,523,551,397]

def test_all_models(signal_array, noise_array):
    methods = ['g', 'sse']

    for method in methods:
        ht = models.HighThreshold(signal_array, noise_array)
        evsd = models.SignalDetection(signal_array, noise_array, equal_variance=True)
        uvsd = models.SignalDetection(signal_array, noise_array, equal_variance=False)
        dpsd = models.DualProcess(signal_array, noise_array)

        ht.fit(method)
        evsd.fit(method)
        uvsd.fit(method)
        dpsd.fit(method)

def test_yonelinas():
    m = models.DualProcess([30, 20, 20, 10, 10, 10], [5, 15, 20, 20, 20, 20])
    m.fit('sse')
    assert np.allclose(m.results['statistic'], 0.00113022192533966, atol=1e-4), f"{m.results['statistic']} different from 0.00113022192533966"

def test_jdcohen_paper():
    dp_low = models.DualProcess(
        [77,18,13,8,4,3,5,4,8,7,4,5,3,7,1,5,5,7,11,5],
        [3,2,8,4,4,3,1,4,11,8,15,15,6,7,6,11,14,28,33,17]
    )
    
    dp_high = models.DualProcess(
        [57,25,13,9,9,3,7,8,13,11,8,11,5,2,4,3,5,6,0,1],
        [5,9,6,8,4,5,7,4,11,12,20,13,10,7,8,5,15,17,28,6]
    )

def test_jdcohen_tutorial1():
    targf = [338, 100, 117, 103, 89, 75, 67, 46, 28, 37]
    luref = [54, 58, 103, 118, 153, 154, 121, 104, 63, 72]
    uvsd = models.SignalDetection(targf, luref, equal_variance=False)
    uvsd.fit()
    
    assert np.allclose(uvsd.fitted_parameters['d'], 1, atol=0.01)
    assert np.allclose(uvsd.fitted_parameters['scale'], 1.4, atol=0.01)
    assert np.allclose(uvsd.LL, -4.2768e3), f"{uvsd.LL} != -4.2768e3"
    assert np.allclose(uvsd.aic, 8.5756e3), f"{uvsd.aic} != 8.5756e3"
    assert np.allclose(uvsd.bic, 8.6372e3), f"{uvsd.bic} != 8.6372e3"


