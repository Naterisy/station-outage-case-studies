import unittest
import numpy as np
from run import predictions

class Predictions(unittest.TestCase):
    def test_exactly_reproduces_drift(self):
        f = np.column_stack((np.ones(5), np.array([1.,4.,5.,10.,11.])))
        y = f @ np.array([2., -.3])
        c = .4 ** np.abs(np.arange(5)[:,None] - np.arange(5))
        np.testing.assert_allclose(predictions(c, f, [0,2,3], [1,4], y[[0,2,3]]), y[[1,4]])

    def test_no_drift_matches_hand_computation(self):
        c = np.array([[2.,.3],[.3,1.]])
        np.testing.assert_allclose(predictions(c, None, [0], [1], np.array([3.])), [.45])

    def test_gls_prediction_matches_augmented_kriging_system(self):
        c = .8 ** np.abs(np.arange(5)[:,None] - np.arange(5)) + .1*np.eye(5)
        f = np.column_stack((np.ones(5),np.arange(5)))
        o, h = [0,2,3], [1,4]
        y = np.array([5.,-.3,2.])
        a = np.block([[c[np.ix_(o,o)],f[o]],[f[o].T,np.zeros((2,2))]])
        b = np.vstack((c[np.ix_(o,h)], f[h].T))
        expected = np.linalg.solve(a,b)[:3].T @ y
        np.testing.assert_allclose(predictions(c, f, o, h, y),expected)
