'''
Test of ABD matrix computation for consistency

Test the ABD matrix for a specific laminate configuration.
'''

import numpy as np
import pytest

from composipy import OrthotropicMaterial, LaminateProperty


@pytest.fixture
def example_laminate():
    E1 = 129500
    E2 = 9370
    v12 = 0.38
    G12 = 5240
    thickness = 0.2
    ply = OrthotropicMaterial(E1, E2, v12, G12, thickness)
    stacking = [90, 0, 90]
    laminate = LaminateProperty(stacking, ply)
    return laminate


def test_abd_matrix(example_laminate):
    expected_abd = np.array([
        [2.99610357e+04, 2.15891656e+03, 1.12894359e-13, -8.52651283e-13, -5.68434189e-14, -1.57772181e-30],
        [2.15891656e+03, 5.42407120e+04, 2.86050843e-12, -5.68434189e-14, -9.09494702e-13, -5.04870979e-29],
        [1.12894359e-13, 2.86050843e-12, 3.14400000e+03, -1.57772181e-30, -5.04870979e-29, -8.52651283e-14],
        [-8.52651283e-13, -5.68434189e-14, -1.57772181e-30, 2.51373035e+02, 6.47674968e+01, 4.89208888e-15],
        [-5.68434189e-14, -9.09494702e-13, -5.04870979e-29, 6.47674968e+01, 2.27467939e+03, 1.23955365e-13],
        [-1.57772181e-30, -5.04870979e-29, -8.52651283e-14, 4.89208888e-15, 1.23955365e-13, 9.43200000e+01]
    ])
    np.testing.assert_allclose(example_laminate.ABD, expected_abd, rtol=1e-2, atol=1e-4)


def test_a_matrix(example_laminate):
    expected_a = np.array([
        [2.99610357e+04, 2.15891656e+03, 1.12894359e-13],
        [2.15891656e+03, 5.42407120e+04, 2.86050843e-12],
        [1.12894359e-13, 2.86050843e-12, 3.14400000e+03]
    ])
    np.testing.assert_allclose(example_laminate.A, expected_a, rtol=1e-2, atol=1e-4)


def test_b_matrix(example_laminate):
    expected_b = np.array([
        [-8.52651283e-13, -5.68434189e-14, -1.57772181e-30],
        [-5.68434189e-14, -9.09494702e-13, -5.04870979e-29],
        [-1.57772181e-30, -5.04870979e-29, -8.52651283e-14]
    ])
    np.testing.assert_allclose(example_laminate.B, expected_b, rtol=1e-2, atol=1e-4)


def test_d_matrix(example_laminate):
    expected_d = np.array([
        [2.51373035e+02, 6.47674968e+01, 4.89208888e-15],
        [6.47674968e+01, 2.27467939e+03, 1.23955365e-13],
        [4.89208888e-15, 1.23955365e-13, 9.43200000e+01]
    ])
    np.testing.assert_allclose(example_laminate.D, expected_d, rtol=1e-2, atol=1e-4)


# Asymmetric cases
@pytest.fixture
def asymmetric_laminates():
    E1 = 129500
    E2 = 9370
    v12 = 0.38
    G12 = 5240
    thickness = 0.2
    ply = OrthotropicMaterial(E1, E2, v12, G12, thickness)
    stackings = [
        [0, 90],
        [45, 0],
        [0, 45, 90],
        [30, -30, 60],
        [0, 45, -45, 90, 30],
        [10, 20, 30, 40, 50],
        [0, 0, 0, 90],
        [45, 45, -45, -45],
        [0, 45, 90, 135],
        [60, 30, 0, -30, -60]
    ]
    laminates = [LaminateProperty(stacking, ply) for stacking in stackings]
    return laminates


def test_abd_asymmetric_0_90(asymmetric_laminates):
    abd = asymmetric_laminates[0].ABD
    expected = np.array([
        [2.81e+04, 1.44e+03, 5.64e-14, -2.43e+03, 0.00e+00, 5.64e-15],
        [1.44e+03, 2.81e+04, 1.43e-12, 0.00e+00, 2.43e+03, 1.43e-13],
        [5.64e-14, 1.43e-12, 2.10e+03, 5.64e-15, 1.43e-13, 0.00e+00],
        [-2.43e+03, 0.00e+00, 5.64e-15, 3.74e+02, 1.92e+01, 7.53e-16],
        [0.00e+00, 2.43e+03, 1.43e-13, 1.92e+01, 3.74e+02, 1.91e-14],
        [5.64e-15, 1.43e-13, 0.00e+00, 7.53e-16, 1.91e-14, 2.79e+01]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_45_0(asymmetric_laminates):
    abd = asymmetric_laminates[1].ABD
    expected = np.array([
        [3.46e+04, 7.05e+03, 6.07e+03, 1.77e+03, -5.61e+02, -6.07e+02],
        [7.05e+03, 1.03e+04, 6.07e+03, -5.61e+02, -6.53e+02, -6.07e+02],
        [6.07e+03, 6.07e+03, 7.70e+03, -6.07e+02, -6.07e+02, -5.61e+02],
        [1.77e+03, -5.61e+02, -6.07e+02, 4.61e+02, 9.40e+01, 8.09e+01],
        [-5.61e+02, -6.53e+02, -6.07e+02, 9.40e+01, 1.38e+02, 8.09e+01],
        [-6.07e+02, -6.07e+02, -5.61e+02, 8.09e+01, 8.09e+01, 1.03e+02]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_0_45_90(asymmetric_laminates):
    abd = asymmetric_laminates[2].ABD
    expected = np.array([
        [3.65e+04, 7.77e+03, 6.07e+03, -4.86e+03, -2.27e-13, -1.73e-13],
        [7.77e+03, 3.65e+04, 6.07e+03, -2.27e-13, 4.86e+03, 1.02e-13],
        [6.07e+03, 6.07e+03, 8.75e+03, -1.73e-13, 1.02e-13, -2.56e-13],
        [-4.86e+03, -2.27e-13, -1.73e-13, 1.24e+03, 8.35e+01, 2.02e+01],
        [-2.27e-13, 4.86e+03, 1.02e-13, 8.35e+01, 1.24e+03, 2.02e+01],
        [-1.73e-13, 1.02e-13, -2.56e-13, 2.02e+01, 2.02e+01, 1.13e+02]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_30_neg30_60(asymmetric_laminates):
    abd = asymmetric_laminates[3].ABD
    expected = np.array([
        [3.56e+04, 1.48e+04, 2.83e+03, -2.43e+03, 4.55e-13, -9.72e+02],
        [1.48e+04, 2.34e+04, 7.69e+03, 2.27e-13, 2.43e+03, 9.72e+02],
        [2.83e+03, 7.69e+03, 1.58e+04, -9.72e+02, 9.72e+02, 0.00e+00],
        [-2.43e+03, 4.55e-13, -9.72e+02, 9.05e+02, 4.43e+02, 4.30e+02],
        [2.27e-13, 2.43e+03, 9.72e+02, 4.43e+02, 8.64e+02, 4.46e+02],
        [-9.72e+02, 9.72e+02, 0.00e+00, 4.30e+02, 4.46e+02, 4.73e+02]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_0_45_neg45_90_30(asymmetric_laminates):
    abd = asymmetric_laminates[4].ABD
    expected = np.array([
        [6.08e+04, 1.90e+04, 7.69e+03, -5.42e+03, 5.61e+02, 1.86e+03],
        [1.90e+04, 4.87e+04, 2.83e+03, 5.61e+02, 4.30e+03, -8.28e+01],
        [7.69e+03, 2.83e+03, 2.07e+04, 1.86e+03, -8.28e+01, 5.61e+02],
        [-5.42e+03, 5.61e+02, 1.86e+03, 7.35e+03, 1.25e+03, 1.50e+03],
        [5.61e+02, 4.30e+03, -8.28e+01, 1.25e+03, 2.45e+03, 7.05e+02],
        [1.86e+03, -8.28e+01, 5.61e+02, 1.50e+03, 7.05e+02, 1.39e+03]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_10_20_30_40_50(asymmetric_laminates):
    abd = asymmetric_laminates[5].ABD
    expected = np.array([
        [7.89e+04, 2.17e+04, 3.02e+04, -9.38e+03, 2.54e+03, 5.11e+02],
        [2.17e+04, 2.53e+04, 1.62e+04, 2.54e+03, 4.31e+03, 3.44e+03],
        [3.02e+04, 1.62e+04, 2.33e+04, 5.11e+02, 3.44e+03, 2.54e+03],
        [-9.38e+03, 2.54e+03, 5.11e+02, 6.54e+03, 1.65e+03, 2.07e+03],
        [2.54e+03, 4.31e+03, 3.44e+03, 1.65e+03, 2.47e+03, 1.45e+03],
        [5.11e+02, 3.44e+03, 2.54e+03, 2.07e+03, 1.45e+03, 1.78e+03]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_0_0_0_90(asymmetric_laminates):
    abd = asymmetric_laminates[6].ABD
    expected = np.array([
        [8.04e+04, 2.88e+03, 5.64e-14, -7.28e+03, 0.00e+00, 1.69e-14],
        [2.88e+03, 3.19e+04, 1.43e-12, 0.00e+00, 7.28e+03, 4.29e-13],
        [5.64e-14, 1.43e-12, 4.19e+03, 1.69e-14, 4.29e-13, 0.00e+00],
        [-7.28e+03, 0.00e+00, 1.69e-14, 3.32e+03, 1.54e+02, 5.27e-15],
        [0.00e+00, 7.28e+03, 4.29e-13, 1.54e+02, 2.67e+03, 1.33e-13],
        [1.69e-14, 4.29e-13, 0.00e+00, 5.27e-15, 1.33e-13, 2.24e+02]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_45_45_neg45_neg45(asymmetric_laminates):
    abd = asymmetric_laminates[7].ABD
    expected = np.array([
        [3.37e+04, 2.53e+04, 0.00e+00, 0.00e+00, 0.00e+00, -4.86e+03],
        [2.53e+04, 3.37e+04, 0.00e+00, 0.00e+00, 0.00e+00, -4.86e+03],
        [0.00e+00, 0.00e+00, 2.66e+04, -4.86e+03, -4.86e+03, 0.00e+00],
        [0.00e+00, 0.00e+00, -4.86e+03, 1.80e+03, 1.35e+03, 0.00e+00],
        [0.00e+00, 0.00e+00, -4.86e+03, 1.35e+03, 1.80e+03, 0.00e+00],
        [-4.86e+03, -4.86e+03, 0.00e+00, 0.00e+00, 0.00e+00, 1.42e+03]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_0_45_90_135(asymmetric_laminates):
    abd = asymmetric_laminates[8].ABD
    expected = np.array([
        [4.49e+04, 1.41e+04, 2.73e-12, -5.98e+03, 1.12e+03, -2.43e+03],
        [1.41e+04, 4.49e+04, -9.09e-13, 1.12e+03, 3.73e+03, -2.43e+03],
        [2.73e-12, -9.09e-13, 1.54e+04, -2.43e+03, -2.43e+03, 1.12e+03],
        [-5.98e+03, 1.12e+03, -2.43e+03, 3.37e+03, 7.52e+02, -4.86e+02],
        [1.12e+03, 3.73e+03, -2.43e+03, 7.52e+02, 1.42e+03, -4.86e+02],
        [-2.43e+03, -2.43e+03, 1.12e+03, -4.86e+02, -4.86e+02, 8.22e+02]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)


def test_abd_asymmetric_60_30_0_neg30_neg60(asymmetric_laminates):
    abd = asymmetric_laminates[9].ABD
    expected = np.array([
        [6.55e+04, 2.04e+04, 4.55e-13, 9.09e-13, -2.27e-13, -5.34e+03],
        [2.04e+04, 4.12e+04, 1.82e-12, -2.27e-13, -9.09e-13, -7.28e+03],
        [9.09e-13, 2.73e-12, 2.21e+04, -5.34e+03, -7.28e+03, 0.00e+00],
        [9.09e-13, -2.27e-13, -5.34e+03, 2.69e+03, 2.04e+03, -1.71e-13],
        [-2.27e-13, -9.09e-13, -7.28e+03, 2.04e+03, 5.52e+03, 0.00e+00],
        [-5.34e+03, -7.28e+03, 0.00e+00, -1.71e-13, 0.00e+00, 2.18e+03]
    ])
    np.testing.assert_allclose(abd, expected, rtol=1e-2, atol=1e-4)
