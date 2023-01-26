import pytest
import numpy as np
from signal_detection import measures
from .context import datasets

@pytest.mark.parametrize('tpr,fpr,expected', [(0.80, 0.40, 1.095), (0.32, 0.04, 1.283), (0.36, 0.36, 0), (.95, .2, 2.486)])
def test_d_prime(tpr, fpr, expected):
    d = measures.d_prime(tpr, fpr)
    assert np.allclose(d, expected, rtol=.01), f"Expected d`={expected} but got d`={d}."

@pytest.mark.parametrize('tpr,fpr,expected', [(.950, .2, -0.402), (.90, .05, 0.182)])
def test_c_bias(tpr, fpr, expected):
    c = measures.c_bias(tpr, fpr)
    assert np.allclose(c, expected, rtol=.01), f"Expected c={expected} but got c={c}."

@pytest.mark.parametrize('tpr,fpr,expected', [(.950, .2, 0.368), (.90, .05, 1.702)])
def test_beta(tpr, fpr, expected):
    beta = measures.beta(tpr, fpr)
    assert np.allclose(beta, expected, rtol=.01), f"Expected β={expected} but got β={beta}."