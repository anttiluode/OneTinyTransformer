"""Quick CPU regression tests; run: python -m unittest discover -s experiments -v"""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from promptless_explorer import load_model
from learned_query_policy import (EpisodeGenerator,Policy,seed_all,split_templates,
                                  log_posteriors,counterfactual_rewards,evaluate)


class LearnedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_all(42)
        cls.m = load_model()

    def test_original_weight_initialization(self):
        p=Policy(self.m,pretrained=True,train_backbone=False)
        np.testing.assert_allclose(p.blocks[0].q.weight.detach().numpy(),self.m['blocks'][0]['wq'],rtol=1e-6,atol=1e-6)
        self.assertFalse(any(x.requires_grad for x in p.blocks.parameters()))

    def test_no_hidden_target_in_policy_inputs(self):
        train,_=split_templates(self.m['embed']); g=EpisodeGenerator(self.m['embed'],train,noise=.9,seed=1)
        tpl,obs,mask,target=g.batch(8)
        policy=Policy(self.m)
        x=policy(tpl,obs,mask,.9)
        self.assertEqual(x.shape,(8,24))
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue((x[mask.bool()] < -1e4).all())
        self.assertTrue(torch.isfinite(counterfactual_rewards(tpl,obs,mask,target,.9)).all())

    def test_disjoint_word_splits(self):
        train,test=split_templates(self.m['embed'])
        self.assertEqual(len(train),45)
        self.assertEqual(len(test),len(self.m['embed'])-1-45)
        self.assertFalse(set(train).intersection(set(test)))

    def test_equal_budget_evaluation(self):
        _,ids=split_templates(self.m['embed']);g=EpisodeGenerator(self.m['embed'],ids,noise=.9,seed=3)
        results=evaluate({},g,.9,n=16,budgets=(1,2,4))
        self.assertEqual(set(results),{'fixed','random','disagreement'})
        for r in results.values():
            for v in r['accuracy'].values():self.assertTrue(0<=v<=1)


if __name__=='__main__':unittest.main()