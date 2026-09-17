import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cropaggregation.aggregation import aggregate,choose_d0,metrics,PC
from cropaggregation.uncertainty import _summary
from cropaggregation.generalization import _aggregate

class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.pixels=pd.DataFrame({
            'polygon_id':[10,10,20,20], 'y_true':[0,0,1,1],
            'distance_px':[0.,2.,0.,0.],
            PC[0]:[.8,.2,.1,.3], PC[1]:[.1,.7,.8,.6], PC[2]:[.1,.1,.1,.1]})

    def test_hard_erosion_drops_unsupported_parcels(self):
        out=aggregate(self.pixels,'M1')
        self.assertEqual(out.polygon_id.tolist(),[10])
        np.testing.assert_allclose(out[PC].iloc[0],[.2,.7,.1])

    def test_soft_weighting_preserves_all_parcels_with_zero_weight_fallback(self):
        out=aggregate(self.pixels,'M2',2)
        self.assertEqual(out.polygon_id.tolist(),[10,20])
        np.testing.assert_allclose(out[PC].to_numpy(),[[.2,.7,.1],[.2,.7,.1]])

    def test_conflicting_labels_rejected(self):
        pixels=self.pixels.copy();pixels.loc[1,'y_true']=1
        with self.assertRaises(ValueError):aggregate(pixels)

    def test_one_se_tie_selects_smallest_candidate(self):
        pixels=self.pixels.copy();pixels['distance_px']=20.
        chosen,_=choose_d0(pixels)
        self.assertEqual(chosen,.5)

    def test_perfect_probabilities_have_zero_proper_score(self):
        pixels=pd.DataFrame({'y_true':[0,1,2],PC[0]:[1,0,0],PC[1]:[0,1,0],PC[2]:[0,0,1]})
        result=metrics(pixels)
        self.assertEqual(result['brier'],0)
        self.assertEqual(result['nll'],0)
        self.assertEqual(result['macro_f1'],1)

    def test_error_ranking_is_directionally_correct(self):
        best=_summary([0,0,1,1],[0.,.1,.8,1.])
        worst=_summary([0,0,1,1],[1.,.8,.1,0.])
        self.assertEqual(best['auroc'],1)
        self.assertLess(best['aurc'],worst['aurc'])

    def test_spatial_and_fixed_weighted_means_agree(self):
        pixels=self.pixels.copy();pixels['distance_px']=[.5,2.,.5,1.]
        pixels.index=[5,8,12,40]
        expected=aggregate(pixels,'M2',2)
        spatial=pixels.rename(columns={'polygon_id':'parcel_id','y_true':'truth',PC[0]:'prob_0',PC[1]:'prob_1',PC[2]:'prob_2'})
        actual=_aggregate(spatial,2)
        np.testing.assert_allclose(actual[['prob_0','prob_1','prob_2']],expected[PC],atol=1e-14)

if __name__=='__main__':unittest.main()
