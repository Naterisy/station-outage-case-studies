import unittest,json
import numpy as np
from run import HERE,fit_season,models,predict
class Tests(unittest.TestCase):
 def test_season_ignores_target_temperatures(self):
  rng=np.random.default_rng(3);past=rng.normal(size=(365,10));spatial=np.c_[np.ones(10),rng.normal(size=(10,3))];train=np.arange(7)
  expected=fit_season(past,train,spatial);past[:,7:]=rng.normal(1e8,1e5,size=(365,3))
  np.testing.assert_array_equal(expected,fit_season(past,train,spatial))
 def test_nested_outer_targets_not_in_inner_prediction_inputs(self):
  folds=np.array([0,1,2,3,4]*2);outer=0;inner=1;available=np.flatnonzero(folds!=outer);train=np.flatnonzero(folds[available]!=inner);hidden=np.flatnonzero(folds[available]==inner)
  y=np.arange(10,dtype=float);C=np.eye(len(available));F=np.ones((len(available),1))
  p=predict(C,F,train,hidden,y[available[train]]);y[folds==outer]=1e10
  np.testing.assert_array_equal(p,predict(C,F,train,hidden,y[available[train]]));self.assertFalse(set(available)&set(np.flatnonzero(folds==outer)))
 def test_covariances_positive_definite(self):
  rng=np.random.default_rng(2);coords=rng.uniform([36,-124],[42,-119],size=(12,2));elev=rng.uniform(0,2000,size=12)
  for grid in models(coords,elev).values():
   for _,C in grid:self.assertGreater(np.linalg.eigvalsh(C).min(),0)
 def test_stratified_folds_and_no_repeated_targets(self):
  f=json.loads((HERE/'folds.json').read_text());self.assertEqual(len({r['station'] for r in f}),36)
  for outer in range(5):self.assertEqual(sum(r['fold']==outer and r['high'] for r in f),2)
if __name__=='__main__':unittest.main()
