# Learned next-observation policy — experiment receipt

**9 November 2026.** Built as a follow-up to [Issue #1](https://github.com/anttiluode/OneTinyTransformer/issues/1) and `promptless_explorer.py`.

## Question

Can a query policy with trainable transformer weights learn which observation would help distinguish competing hidden patterns, without being hardcoded to maximize candidate variance? Does initializing from this repo's pretrained tiny language model help?

## Procedure

The hidden target is one of eight candidate 24-dimensional word embeddings. The observer knows all eight *templates*, but not which one is active. A coordinate observation returns the hidden target's value with Gaussian noise σ=0.9. A common, ideal Gaussian decoder updates a posterior over the eight candidates after each measurement; the competing policies only differ in **which unobserved coordinate to inspect next**. Targets are chosen uniformly; the candidate shortlist comprises the nearest eight embeddings to a randomly selected anchor within the split.

The model uses 24 coordinate tokens, each exposing eight candidate values, eight posterior probabilities, its current observed value (zero when unobserved), and the observation mask. There are **no oracle target labels or unobserved values in the policy input**. This new input adapter feeds a transformer with four blocks, two heads, dimension 24, and the original RoPE scheme. Bidirectional attention across candidate coordinates is permitted because all templates are already visible. The block weights are copied from the original `index.html` and either tuned or frozen; a randomly initialized, trainable control has the same structure.

During training, the simulated environment supplies the target identity and the counterfactual observations for every eligible next coordinate. The optimization objective maximizes the *actual change in log posterior probability of the hidden target* over these actions; it does not train on the analytic maximum-variance choice. This is **privileged training feedback** and cannot be assumed available in an unlabelled real-world task. No extra labels are supplied during evaluation.

Seed 20261109, training 300 minibatches of 96 for each variant, word split 45 train / 24 held-out, evaluation 960 new held-out trials. Deterministic test setup, with all policies evaluated on the same targets and noisy observations. Computation cost per decision is not matched: the analytic variance selector is much cheaper than the four-block policy. No multi-seed confidence interval was run.

## Results

| Policy | 1 measurement | 2 | 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 28.5% | 39.0% | 60.7% | 74.0% | 84.7% | 93.7% |
| Random | 26.4% | 40.4% | 62.5% | 76.8% | 85.1% | 93.3% |
| Analytic disagreement | 35.3% | 59.1% | **87.8%** | 96.5% | 97.7% | 99.5% |
| Pretrained, tuned | 32.0% | 50.7% | **73.0%** | 84.1% | 89.4% | 97.1% |
| Pretrained, frozen | 34.1% | 48.7% | 71.9% | 84.5% | 91.7% | 96.9% |
| Random initialization, tuned | 35.2% | 50.7% | 72.6% | 84.2% | 92.1% | 97.1% |

Source: `results_learned_policy.json`; these are **observed results**, not predictions.

## Interpretation

**Positive:** policy training with feedback improves four-look accuracy by 10.5 percentage points versus random querying and 12.3 points versus fixed querying on these held-out trials.

**Negative:** the tuned pretrained model is only 0.4 points ahead of the random-initialization control, without repetitions to establish whether the difference is meaningful. The analytic variance rule remains 14.8 points ahead. These results do **not** establish that an ordinary pretrained language transformer learned intrinsic curiosity, or that pretrained LM features are essential to effective active sensing.

**Scientific boundary:** the Gaussian decoder already supplies a highly useful posterior sufficient statistic. The learned transformer chooses observations; it does not learn the posterior or invent the external question. The result is not a promptless language model thinking autonomously. No claim about actual perception, general image search, or cross-level biological causation follows.

## Reproduction

```bash
pip install numpy torch
python experiments/learned_query_policy.py --steps 300 --trials 960
python -m unittest discover -s experiments -t . -v
```

Run from the repository root. No Internet or GPU. The local checkpoint `learned_policy.pt` is an output artifact, not required to reproduce training.

## Next falsifiable gate

Can the policy remain useful when posterior probabilities are **not supplied** and must be inferred from raw observations, or when the observation noise/model is misspecified? Repeat across multiple seeds and harder held-out candidate families. Do not interpret visually complex recurrent trajectories as successful information acquisition without a scored task.