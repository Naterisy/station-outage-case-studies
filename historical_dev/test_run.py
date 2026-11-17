import unittest,json
import numpy as np
import pandas as pd
from run import predictions,HERE
class HistoricalTests(unittest.TestCase):
 def test_no_hidden_response_input(self):
  cov=np.array([[2.,1.,.5],[1.,2.,.8],[.5,.8,2.]])
  pred=predictions(cov,None,np.array([0,1]),np.array([2]),np.array([3.,4.]))
  np.testing.assert_allclose(pred,cov[2,:2]@np.linalg.solve(cov[:2,:2],[3.,4.]))
 def test_covariance_training_reconstruction(self):
  with np.load(HERE.parent/'realdata/dev/2021.npz') as z: past=z['temperature_c']
  t=np.arange(365)*2*np.pi/365;X=np.c_[np.ones(365),np.sin(t),np.cos(t),np.sin(2*t),np.cos(2*t)]
  residual=past.copy()
  for j in range(past.shape[1]):
   ok=np.isfinite(past[:,j]);residual[:,j]-=X@np.linalg.lstsq(X[ok],past[ok,j],rcond=None)[0]
  ok=np.isfinite(residual).all(1)
  with np.load(HERE/'training_covariance.npz') as z:
   np.testing.assert_allclose(z['covariance'],np.cov(residual[ok],rowvar=False))
   np.testing.assert_array_equal(z['complete_days'],np.flatnonzero(ok))
   self.assertGreater(np.linalg.eigvalsh(z['covariance']).min(),0)
 def test_same_original_predictions_and_counts(self):
  old=pd.read_csv(HERE.parent/'kriging_dev/summary.csv').set_index(['mask','method'])
  new=pd.read_csv(HERE/'summary.csv').set_index(['mask','method'])
  for mask in ['random','blocked']:
   for suffix in ['fixed','drift']:
    np.testing.assert_allclose(old.loc[(mask,'spatial_elev_'+suffix),['n','mae','p90','high_mae']].astype(float),new.loc[(mask,'original_geographic_'+suffix),['n','mae','p90','high_mae']].astype(float),atol=1e-12)
 def test_disjoint_week_partition(self):
  c=json.loads((HERE/'cases.json').read_text())
  for cases in c.values():
   for day,o,h in cases:self.assertFalse(set(o)&set(h))
   tune={day for day,_,_ in cases if day//7%2==0};diag={day for day,_,_ in cases if day//7%2==1}
   self.assertFalse(tune&diag);self.assertEqual(len(tune|diag),365)
if __name__=='__main__':unittest.main()
