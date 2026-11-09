"""Frozen OneTinyTransformer autonomous query + active-sensing experiment.

Run from repo root: python experiments/promptless_explorer.py
Dependency: numpy. Reads the exact MODEL embedded in ../index.html. No training.
"""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = [11, 23, 37, 51, 67, 83]
STEPS = 96
CANDIDATES = 6


def load_model(path=ROOT / 'index.html'):
    html = path.read_text(encoding='utf-8')
    match = re.search(r'^const MODEL = (\{.*\});$', html, re.MULTILINE)
    if not match:
        raise ValueError('Could not find inline trained MODEL in index.html')
    data = json.loads(match.group(1))
    d = data['meta']['d']
    model = {
        'meta': data['meta'],
        'embed': np.array(data['embed'], dtype=np.float64).reshape(-1, d),
        'out': np.array(data['out'], dtype=np.float64).reshape(-1, d),
        'blocks': [],
    }
    for block in data['blocks']:
        model['blocks'].append({
            name: np.array(vals, dtype=np.float64).reshape(
                data['meta']['ff'] if name == 'fc1' else
                d if name in ('wq', 'wk', 'wv', 'wo', 'fc2') else -1,
                d if name in ('wq', 'wk', 'wv', 'wo') else
                data['meta']['ff'] if name == 'fc2' else -1,
            )
            for name, vals in block.items()
        })
    return model


def layernorm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)


def gelu(x):
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))


def rope(x, base):
    # (T, heads, dh); original JS pairs adjacent features.
    t, h, dh = x.shape
    a = np.arange(t, dtype=float)[:, None, None] * (base ** (-2 * np.arange(dh//2) / dh))[None, None, :]
    c, s = np.cos(a), np.sin(a)
    out = x.copy()
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
    return out


def forward(m, latent_history, return_heads=False):
    """Exact causal math and weight layout of index.html, but arbitrary latent inputs."""
    meta = m['meta']; H = meta['heads']; dh = meta['dh']; d = meta['d']
    x = np.asarray(latent_history, dtype=np.float64)
    n = len(x)
    assert 1 <= n <= meta['context'] and x.shape == (n, d)
    heads_last = []
    for b in m['blocks']:
        h = layernorm(x)
        q = rope((h @ b['wq'].T).reshape(n, H, dh), meta['base'])
        k = rope((h @ b['wk'].T).reshape(n, H, dh), meta['base'])
        v = (h @ b['wv'].T).reshape(n, H, dh)
        scores = np.einsum('thd,shd->hts', q, k) / math.sqrt(dh)
        scores[:, np.triu_indices(n, 1)[0], np.triu_indices(n, 1)[1]] = -np.inf
        scores -= np.max(scores, axis=2, keepdims=True)
        p = np.exp(scores); p /= np.sum(p, axis=2, keepdims=True)
        if return_heads: heads_last.extend([p[head, -1].copy() for head in range(H)])
        gathered = np.einsum('hts,shd->thd', p, v).reshape(n, d)
        mid = x + gathered @ b['wo'].T
        x = mid + gelu(layernorm(mid) @ b['fc1'].T) @ b['fc2'].T
    return (x[-1], heads_last) if return_heads else x[-1]


def unit(x):
    return x / max(1e-12, np.linalg.norm(x))


def attention_disagreement(heads):
    """Jensen-Shannon divergence of heads over prior positions; not confidence."""
    a = np.array(heads)
    mean = a.mean(axis=0)
    return float(np.mean(np.sum(a * (np.log(np.maximum(a, 1e-15)) - np.log(np.maximum(mean, 1e-15))), axis=1)))


def softmax(a):
    z = np.exp(a - np.max(a)); return z / z.sum()


def explore(m, seed, policy):
    rng = np.random.default_rng(seed)
    e = m['embed']; words = m['meta']['vocab']; context = m['meta']['context']
    base_norm = np.linalg.norm(e, axis=1).mean()
    # No text prompt. Seed latent state is a reproducible, random initial condition.
    z0 = unit(e.mean(axis=0) + .2 * rng.standard_normal(e.shape[1])) * base_norm
    state = [z0]
    chosen, entropy, similarity, disagreement, trace = [], [], [], [], []
    for t in range(STEPS):
        history = np.array(state[-context:])
        h, heads = forward(m, history, return_heads=True)
        probs = softmax((m['out'] @ layernorm(h)) / .75)
        probs[0] = 0.; probs /= probs.sum()
        entropy.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-15)))))
        disagreement.append(attention_disagreement(heads))
        ranked = np.argsort(-probs)
        if policy == 'replay':
            next_z = unit(.5 * unit(state[-1]) + .5 * unit(h)) * base_norm
            tag = None
        elif policy == 'greedy':
            tag = int(ranked[0]); next_z = e[tag].copy()
        elif policy == 'random':
            tag = int(rng.choice(ranked[:CANDIDATES])); next_z = e[tag].copy()
        elif policy == 'disagreement':
            options = []
            for idx in ranked[:CANDIDATES]:
                seq = np.array((state + [e[idx]])[-context:])
                _, ah = forward(m, seq, return_heads=True)
                options.append(attention_disagreement(ah))
            tag = int(ranked[int(np.argmax(options))]); next_z = e[tag].copy()
        else:
            raise ValueError(policy)
        similarity.append(float(np.dot(unit(state[-1]), unit(next_z))))
        chosen.append(tag)
        if t % 8 == 0: trace.append({'step':t, 'top_word':words[int(ranked[0])], 'selected':words[tag] if tag is not None else None,
                                       'entropy':round(entropy[-1], 4), 'jsd':round(disagreement[-1], 4)})
        state.append(next_z)
    # Exact embeddings imply an exact symbolic cycle, if one exists.
    codes = chosen if policy != 'replay' else [tuple(np.round(z, 4)) for z in state[1:]]
    cycle = None
    for period in range(1, 33):
        if codes[-3*period:] == codes[-period:]*3:
            cycle = period
            break
    return {'seed': seed, 'policy':policy, 'steps':STEPS,
            'unique_selected':len(set(codes)), 'last32_unique':len(set(codes[-32:])),
            'cycle_period_le32':cycle, 'mean_logit_entropy':round(float(np.mean(entropy)),4),
            'mean_attention_jsd':round(float(np.mean(disagreement)),4),
            'mean_consecutive_cosine':round(float(np.mean(similarity)),4), 'trace_every8':trace,
            'selected_sequence':chosen}


def active_sensing(m, trials=480, noise=0.55, budget=12, seed=20261109):
    """Controlled information acquisition using frozen embeddings as hypotheses.

    A shortlist is selected before the target, from nearest neighbours of a random anchor.
    All policies see identical pre-generated Gaussian noisy coordinate observations.
    """
    rng = np.random.default_rng(seed)
    e = m['embed'][1:]  # exclude <unk>
    cos = e / np.linalg.norm(e,axis=1,keepdims=True)
    scores = {p:{b:0 for b in ((2,4,6,8,12,24) if p=='full24' else (2,4,6,8,12))} for p in ('fixed','random','disagreement','full24')}
    sample_count = 0
    for _ in range(trials):
        anchor = rng.integers(len(e))
        ids = np.argsort(-(cos @ cos[anchor]))[:8]
        target = int(rng.integers(len(ids))) # uniformly selected AFTER the shortlist
        templates = e[ids]
        obs = templates[target] + rng.normal(0, noise, e.shape[1])
        post_by_policy = {}
        for policy in ('fixed','random','disagreement','full24'):
            remaining = list(range(e.shape[1])); logp = np.zeros(len(ids),dtype=float)
            path=[]
            steps = 24 if policy=='full24' else budget
            for look in range(steps):
                p = softmax(logp)
                if policy in ('fixed','full24'): ix = remaining[0]
                elif policy == 'random': ix = int(rng.choice(remaining))
                else:
                    # posterior-weighted predictive variance over an UNOBSERVED coordinate
                    mu = p @ templates[:, remaining]
                    var = p @ (templates[:, remaining] - mu)**2
                    ix = remaining[int(np.argmax(var))]
                remaining.remove(ix)
                logp += -.5*((obs[ix]-templates[:,ix])/noise)**2
                path.append(int(np.argmax(logp) == target))
            post_by_policy[policy] = path
        for p, path in post_by_policy.items():
            for b in scores[p]:
                scores[p][b] += path[b-1] if b <= len(path) else path[-1]
        sample_count += 1
    return {'trials':sample_count, 'noise_sd':noise, 'candidate_count':8,
            'shortlist':'nearest eight to random anchor; target uniform afterward',
            'identical_observation_noise_across_policies':True,
            'accuracy_at_observations':{p:{str(b):round(v/sample_count,4) for b,v in tally.items()} for p,tally in scores.items()},
            'note':'full24 is fixed coordinate order, including a 24-observation ceiling; its shorter prefixes are identical to fixed.'}


def summarize_runs(runs):
    out={}
    for name in ('replay','greedy','random','disagreement'):
        subset=[r for r in runs if r['policy']==name]
        out[name]={
            'last32_unique_mean':round(float(np.mean([r['last32_unique'] for r in subset])),2),
            'unique_selected_mean':round(float(np.mean([r['unique_selected'] for r in subset])),2),
            'mean_logit_entropy':round(float(np.mean([r['mean_logit_entropy'] for r in subset])),3),
            'mean_attention_jsd':round(float(np.mean([r['mean_attention_jsd'] for r in subset])),3),
            'cycle_le32_seeds':sum(r['cycle_period_le32'] is not None for r in subset),
            'cycle_periods':[r['cycle_period_le32'] for r in subset],
        }
    return out


def main():
    m=load_model(); assert m['meta']['d']==24 and m['meta']['layers']==4
    runs=[explore(m,seed,policy) for seed in SEEDS for policy in ('replay','greedy','random','disagreement')]
    # Destroy the learned pairing between the output head and input embeddings.
    controls = {}
    for label, key in [('unpaired_output', 'out'), ('unpaired_embeddings', 'embed')]:
        disrupted = dict(m)
        rng = np.random.default_rng(456)
        disrupted[key] = m[key][rng.permutation(len(m[key]))]
        control_runs = [explore(disrupted, s, policy) for s in SEEDS for policy in ('greedy', 'disagreement')]
        controls[label] = {p: {'cycles': [x['cycle_period_le32'] for x in control_runs if x['policy']==p],
                               'last32_unique_mean': round(float(np.mean([x['last32_unique'] for x in control_runs if x['policy']==p])),2)}
                           for p in ('greedy','disagreement')}
    synthetic = dict(m)
    rng = np.random.default_rng(456)
    synthetic['embed'] = rng.normal(size=m['embed'].shape)*m['embed'].std()
    sensing_control = active_sensing(synthetic)
    result={'model':'Original OneTinyTransformer trained weights from index.html',
            'method':'fresh causal forward pass each step; up to 24 latent states as virtual tokens; no external text prompt',
            'autonomous':{'seeds':SEEDS,'steps':STEPS,'candidate_count':CANDIDATES,'runs':runs,'summary':summarize_runs(runs)},
            'active_sensing':active_sensing(m),
            'control_autonomous':controls,
            'control_sensing_random_vectors':sensing_control}
    dest=HERE/'results_promptless.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print('AUTONOMOUS',json.dumps(result['autonomous']['summary'],indent=2))
    print('ACTIVE SENSING',json.dumps(result['active_sensing'],indent=2))
    print('CONTROLS',json.dumps(controls,indent=2))
    print('RANDOM VECTOR CONTROL 4 OBS',{p:v['4'] for p,v in sensing_control['accuracy_at_observations'].items()})
    summary = {k:v for k,v in result.items() if k!='autonomous'}
    summary['autonomous'] = {k:v for k,v in result['autonomous'].items() if k!='runs'}
    (HERE/'results_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Saved',dest)


if __name__=='__main__': main()
