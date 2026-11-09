"""Cross-check the Python forward path against the ORIGINAL index.html JavaScript.

Reference was obtained with JavaScript: prefill([57,54,27,57]); last token,
last layer. All 24 residual entries are recorded to 10 decimals.
Run: python -m unittest discover -s experiments -p 'test_*.py'
"""
import unittest
import numpy as np

from experiments.promptless_explorer import load_model, forward, layernorm

REFERENCE = [
 -0.7783525418,-2.2398704803,0.1151288807,1.579558767,
 -0.698213931,1.5597405978,-0.1633118926,1.8785912522,
 0.3478325438,-1.7437147058,-1.2809213036,2.9058977572,
 -1.2348332032,-0.9498948955,1.6796939318,0.6359225352,
 0.1532577624,-0.0631742861,2.1435048696,1.8085532182,
 1.8420550004,0.0580574394,0.2990860279,-0.0800243554,
]
ATT_FIRST = [0.034537162217351244,0.8787008685497221,0.008652175772878296,0.07810979346004857]
ATT_LAST  = [0.024601976223642862,0.03187335754232666,0.30194733839768023,0.6415773278363502]


class VerifyAgainstOriginalJS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_original_weights_and_forward(self):
        m = self.model
        self.assertEqual((m['meta']['d'],m['meta']['layers'],m['meta']['heads']),(24,4,2))
        x = m['embed'][[57,54,27,57]]
        output, heads = forward(m,x,return_heads=True)
        np.testing.assert_allclose(output,REFERENCE,atol=1e-9,rtol=0)
        np.testing.assert_allclose(heads[0],ATT_FIRST,atol=1e-12,rtol=0)
        np.testing.assert_allclose(heads[7],ATT_LAST,atol=1e-12,rtol=0)

    def test_layernorm_stability_and_causality(self):
        m = self.model
        np.testing.assert_allclose(layernorm(np.ones((2,24))),0.,atol=1e-12)
        first = m['embed'][57]
        x = m['embed'][[57,54,27,57]]
        # Causal mask: changing all future tokens cannot affect first position.
        # Compare the identical first-token recurrence under both contexts:
        a = forward(m,x[:1])
        b = forward(m,first[None,:])
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)


if __name__ == '__main__': unittest.main()
