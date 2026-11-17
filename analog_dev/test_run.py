import unittest
import numpy as np
import pandas as pd
from run import HERE,analogs
class Tests(unittest.TestCase):
 def test_rank_uses_observed_coordinates_only(self):
  r=np.random.default_rng(1).normal(size=(100,5));o=np.array([0,2,4]);h=np.array([1,3]);v=np.array([1.,2.,3.]);B=np.zeros((2,3))
  _,rank=analogs(r,v,o,h,np.ones(5),B);changed=r.copy();changed[:,h]*=1000
  _,rank2=analogs(changed,v,o,h,np.ones(5),B);np.testing.assert_array_equal(rank,rank2)
 def test_corrected_equals_covariance_plus_reference_residual(self):
  r=np.random.default_rng(2).normal(size=(100,5));o=np.array([0,2,4]);h=np.array([1,3]);v=np.array([1.,2.,3.]);B=np.ones((2,3))*.1
  pred,rank=analogs(r,v,o,h,np.ones(5),B);a=r[rank[:5]].mean(0)
  np.testing.assert_allclose(pred[5][2],B@v+a[h]-B@a[o])
 def test_empirical_reproduces_historical(self):
  old=pd.read_csv(HERE.parent/'historical_dev/summary.csv').set_index(['mask','method']);new=pd.read_csv(HERE/'summary.csv').set_index(['mask','method'])
  for mask in ['random','blocked']:
   np.testing.assert_allclose(old.loc[(mask,'empirical_fixed'),['n','mae','p90','high_mae']].astype(float),new.loc[(mask,'empirical'),['n','mae','p90','high_mae']].astype(float),atol=1e-12)
 def test_references_only_complete_training_days(self):
  with np.load(HERE.parent/'realdata/dev/2021.npz') as z: x=z['temperature_c']
  np.testing.assert_array_equal(np.load(HERE/'reference_day_indices.npy'),np.flatnonzero(np.isfinite(x).all(1)))
if __name__=='__main__':unittest.main()
