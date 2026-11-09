# Frozen transformer, self-generated queries — experiment (9 Nov 2026)

Research question: [Issue #1](https://github.com/anttiluode/OneTinyTransformer/issues/1).

**Outcome:** A frozen, trained transformer can be placed inside a recurrence without a human text prompt, but our first disagreement-based rule often settles into a 3-state cycle. A *separate* active-sensing task shows that choosing statistically informative vector coordinates beats fixed/random measurement. Its advantage also appears with random vectors and must **not** be attributed to transformer intelligence.

## Reproduction

```bash
pip install numpy
python experiments/promptless_explorer.py
```

Uses the original model embedded in `index.html` (24-D embedding/residual, four layers, two heads, RoPE, 70-word vocabulary, context 24), with *no training*. A reference four-token forward pass agreed with the original JavaScript to 1.11e-15 max absolute residual error and 3.89e-16 max attention error. Six deterministic initial states (seeds 11, 23, 37, 51, 67, 83), 96 recurrent steps each. No text prompt, but a seeded latent starting state is necessary.

## Experiment A — frozen latent recurrence

Each new virtual token is processed by the original four transformer blocks; prior states are kept in a sliding causal window of at most 24. Greedy chooses the top-scored vocabulary embedding; random picks uniformly from the top six; disagreement hypothetically appends each top-six embedding and selects the one that maximizes Jensen–Shannon divergence among the eight attention heads' distributions over prior positions. Replay instead feeds a normalized continuous blend of the old state and the model's resulting latent state. All matrices stay frozen.

| Policy | Mean distinct selected states in final 32 steps | Exact final cycle (period ≤32) | Mean head JSD |
|---|---:|---:|---:|
| Continuous replay | 32.00 (rounded to 4dp) | 0/6 seeds | 0.423 |
| Greedy top choice | 14.17 | 0/6 | 0.669 |
| Uniform random, top 6 | 19.50 | 1/6 | 0.543 |
| Max head disagreement | **8.17** | **4/6 (period 3)** | **0.792** |

**Interpretation:** higher head disagreement did **not** imply exploration. The recurrence frequently entered short cycles; the chosen scoring rule can reward an attractor. Heads are not calibrated independent hypotheses.

**Weight pairing controls:** Permuting output rows while leaving embeddings unchanged caused greedy to fall to a fixed point in all six seeds, whereas the disagreement policy had no detected ≤32 cycle. Permuting the embeddings made greedy much more repetitive and disagreement end in period-1 cycles on two seeds. This shows the trained input/output pairing affects dynamics, not that the loops encode useful concepts.

## Experiment B — statistical choice of where to look

A separate controlled observation experiment uses only the frozen 24-D input embedding table. From eight nearby candidate embeddings (chosen before hiding the target), choose a target uniformly. At each look reveal one previously unseen coordinate with Gaussian noise SD 0.55. Exact posterior over eight candidates; compare fixed order, random coordinates, and posterior-variance maximum. The same pre-generated observation noise is shared across policies. 480 trials.

| Measurements | Fixed | Random | Posterior disagreement |
|---:|---:|---:|---:|
| 2 | 53.96% | 51.88% | **76.04%** |
| 4 | 78.75% | 77.50% | **98.75%** |
| 6 | 90.62% | 92.92% | **100.00%** |
| 8 | 96.46% | 97.50% | **100.00%** |
| 12 | 99.38% | 99.58% | **100.00%** |

With all 24 coordinates, fixed-order full-scan accuracy was 100% on these trials.

**Vital negative control:** Replacing learned embeddings by independent Gaussian vectors yields **98.75% adaptive vs 79.37% fixed and 83.33% random** at four measurements. This is a generic Bayesian active-measurement effect, not evidence that transformer attention learned a curiosity mechanism.

## What was and wasn't learned

The model **wasn't trained for recurrent self-feeding**. Experiment A's recurrence and policy were engineered for the test. Experiment B does not use the transformer blocks at all; it uses the embedding vectors as a hypothesis bank. No new external information can appear from autonomous dynamics alone.

The next discriminating gate is to feed *partial noisy evidence* into the transformer, ask its **actual attention computations** to propose the next probe, and score that policy on a held-out identification task against random, fixed, and posterior-variance oracle under identical measurement **and compute** budgets. If it cannot beat random, stop.

This distinction—**an internal attractor** versus **an adaptive sensor whose questions yield information**—is the central experimental result.
