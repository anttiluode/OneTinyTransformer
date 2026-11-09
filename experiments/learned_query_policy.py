"""Train/test active query selection using the actual OneTinyTransformer weights.

Runs entirely on CPU. The query policy is a 24-token transformer initialized
from the existing language model's 4 attention/MLP blocks; an input adaptor
and an action head are trained, and optionally the copied blocks themselves.

IMPORTANT: This is *not* an intrinsic behavior of the original language model.
The coordinate-query interface, reward and training are newly defined here.
Training rewards come from simulated observations of a hidden target, not from
imitating the analytic posterior-variance ('disagreement') policy.

Run from the repository root:
    python experiments/learned_query_policy.py --steps 500
Dependencies: numpy, torch. Uses no GPU, downloads or external dataset.
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from promptless_explorer import load_model

HERE = Path(__file__).resolve().parent
SEED = 20261109
NC = 8
D = 24


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def split_templates(embed):
    """Disjoint word embeddings; no lexical identity leaks across splits."""
    rng = np.random.default_rng(9917)
    ids = rng.permutation(np.arange(1, len(embed)))
    return ids[:45], ids[45:]


class EpisodeGenerator:
    def __init__(self, embed, ids, noise=.9, seed=1):
        self.e = np.asarray(embed, np.float32)[ids]
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        nrm = self.e / np.linalg.norm(self.e, axis=1, keepdims=True)
        self.neighbours = np.argsort(-(nrm @ nrm.T), axis=1)[:, :NC]

    def batch(self, n, min_prior=0, max_prior=5):
        rng = self.rng
        anchors = rng.integers(0, len(self.e), size=n)
        group = self.neighbours[anchors].copy()
        # Shuffle candidate positions independently, otherwise the anchor slot is special.
        shuffles = np.argsort(rng.random((n, NC)), axis=1)
        group = np.take_along_axis(group, shuffles, axis=1)
        tpl = self.e[group]
        targets = rng.integers(0, NC, size=n)
        noisy = tpl[np.arange(n), targets] + rng.normal(0, self.noise, (n, D))
        prior = rng.integers(min_prior, max_prior + 1, size=n)
        order = np.argsort(rng.random((n, D)), axis=1)
        mask = (np.argsort(order, axis=1) < prior[:, None]).astype(np.float32)
        return (torch.tensor(tpl), torch.tensor(noisy, dtype=torch.float32),
                torch.tensor(mask), torch.tensor(targets, dtype=torch.long))


def log_posteriors(tpl, observed, mask, noise):
    """Shared ideal Gaussian decoder across all policies. It never chooses actions."""
    sq = (tpl - observed[:, None, :]).square()
    logits = -(mask[:, None, :] * sq).sum(dim=-1) / (2 * noise * noise)
    return F.log_softmax(logits, dim=-1)


def counterfactual_rewards(tpl, observed, mask, targets, noise):
    """Environment feedback: change in log probability of actual hidden target.

    During *training only*, every candidate action gets its own counterfactual
    outcome using the held-out measured value at that coordinate. The policy
    never observes the true target or these outcomes at evaluation time.
    """
    prev = log_posteriors(tpl, observed, mask, noise)
    sq = (tpl - observed[:, None, :]).square()
    scores = prev[:, :, None] - sq / (2 * noise * noise)
    updated = scores - torch.logsumexp(scores, dim=1, keepdim=True)
    true_new = updated.gather(1, targets[:, None, None].expand(-1, 1, D)).squeeze(1)
    true_prev = prev.gather(1, targets[:, None]).squeeze(1)
    return (true_new - true_prev[:, None]).clamp(-7, 7).masked_fill(mask.bool(), 0)


class ReusedBlock(nn.Module):
    def __init__(self, weights, use_pretrained=True):
        super().__init__()
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.fc1 = nn.Linear(D, 48, bias=False)
        self.fc2 = nn.Linear(48, D, bias=False)
        if use_pretrained:
            with torch.no_grad():
                for nm, key in [('q','wq'),('k','wk'),('v','wv'),('o','wo'),('fc1','fc1'),('fc2','fc2')]:
                    getattr(self, nm).weight.copy_(torch.tensor(weights[key], dtype=torch.float32))

    def forward(self, x):
        # Non-causal attention: *all* candidate coordinate values are known.
        # Querying a hidden observation value is still prohibited by the mask.
        B, T, _ = x.shape
        h = F.layer_norm(x, (D,))
        q = self.q(h).reshape(B, T, 2, 12).transpose(1, 2)
        k = self.k(h).reshape(B, T, 2, 12).transpose(1, 2)
        v = self.v(h).reshape(B, T, 2, 12).transpose(1, 2)
        # The original RoPE position parameters, applied to 24 coordinate tokens.
        j = torch.arange(6, device=x.device, dtype=x.dtype)
        freq = torch.pow(torch.tensor(100.,device=x.device), -2*j/12)
        th = torch.arange(T, device=x.device, dtype=x.dtype)[:, None] * freq[None, :]
        c, s = th.cos()[None, None], th.sin()[None, None]
        def rope(a):
            xx = a.reshape(B, 2, T, 6, 2)
            r = torch.stack((xx[..., 0]*c-xx[..., 1]*s,xx[..., 0]*s+xx[..., 1]*c),dim=-1)
            return r.reshape(B, 2, T, 12)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=False)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        g = F.layer_norm(x, (D,))
        # Match the JS implementation's tanh-approx GELU.
        z = self.fc1(g)
        z = .5*z*(1 + torch.tanh(math.sqrt(2/math.pi)*(z + .044715*z.pow(3))))
        return x + self.fc2(z)


class Policy(nn.Module):
    def __init__(self, m, pretrained=True, train_backbone=True):
        super().__init__()
        # 8 raw candidate values; 8 current posterior masses; observed value,
        # observed mask. No analytic variance, information gain or oracle score.
        self.input = nn.Linear(18, D)
        self.blocks = nn.ModuleList([ReusedBlock(b, pretrained) for b in m['blocks']])
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(),nn.Linear(D, 1))
        if not train_backbone:
            for p in self.blocks.parameters(): p.requires_grad_(False)
        self.pretrained = pretrained
        self.train_backbone = train_backbone

    def forward(self, tpl, observed, mask, noise):
        p = log_posteriors(tpl, observed, mask, noise).exp()
        feat = torch.cat((tpl.transpose(1, 2), p[:,None,:].expand(-1,D,-1),
                          observed[:, :, None] * mask[:, :, None], mask[:, :, None]),dim=-1)
        h = self.input(feat)
        for block in self.blocks: h = block(h)
        logits = self.head(h).squeeze(-1)
        return logits.masked_fill(mask.bool(), -1e5)


def train_policy(m, gen, noise, steps=500, pretrained=True, train_backbone=True, seed=77, batch_size=96):
    seed_all(seed)
    policy = Policy(m, pretrained, train_backbone)
    opt = torch.optim.AdamW([
        {'params':list(policy.input.parameters())+list(policy.head.parameters()),'lr':0.0015},
        {'params':[p for p in policy.blocks.parameters() if p.requires_grad], 'lr':0.00025},
    ],weight_decay=0.0005)
    trace=[]
    t0=time.time()
    for step in range(steps):
        tpl, obs, mask, targets = gen.batch(batch_size)
        reward = counterfactual_rewards(tpl,obs,mask,targets,noise)
        logits = policy(tpl, obs, mask, noise)
        prob = logits.softmax(dim=-1)
        # All-action counterfactual policy gradient (not imitation of the
        # variance rule). Center baseline to lower variance.
        r = (reward - (reward * (~mask.bool())).sum(dim=1,keepdim=True) / (~mask.bool()).sum(dim=1,keepdim=True))
        loss = -(prob * r.detach()).sum(dim=1).mean() - .002 * (-(prob*prob.clamp_min(1e-8).log()).sum(1)).mean()
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.0);opt.step()
        if step in (0, steps//4, steps//2, 3*steps//4, steps-1):
            trace.append({'step':step+1, 'loss':round(loss.item(),5),'sec':round(time.time()-t0,1)})
    return policy.eval(),trace


@torch.no_grad()
def evaluate(models, gen, noise, n=480, budgets=(1,2,4,6,8,12), chunk_size=128):
    """Paired evaluation in small CPU batches; all policies see identical trials."""
    policies = ['fixed','random','disagreement'] + list(models)
    totals = {name:{'correct':{str(b):0 for b in budgets},
                    'logp':{str(b):0. for b in budgets},'first_choices':[]}
              for name in policies}
    rnd = torch.Generator().manual_seed(SEED + 151)
    done=0
    while done < n:
        tpl,obs,_,targets=gen.batch(min(chunk_size,n-done),0,0)
        B=len(tpl);idx=torch.arange(B)
        for name in policies:
            mask=torch.zeros((B,D))
            for t in range(max(budgets)):
                if name=='fixed':
                    choice=(~mask.bool()).int().argmax(dim=1)
                elif name=='random':
                    raw=torch.rand((B,D),generator=rnd).masked_fill(mask.bool(),-1)
                    choice=raw.argmax(dim=1)
                elif name=='disagreement':
                    p=log_posteriors(tpl,obs,mask,noise).exp()
                    mu=(tpl*p[:,:,None]).sum(1)
                    var=((tpl-mu[:,None,:]).square()*p[:,:,None]).sum(1)
                    choice=var.masked_fill(mask.bool(),-1).argmax(1)
                else:
                    choice=models[name](tpl,obs,mask,noise).argmax(1)
                mask[idx,choice]=1
                if t==0 and len(totals[name]['first_choices']) < 12:
                    totals[name]['first_choices'] += choice[:12-len(totals[name]['first_choices'])].tolist()
                if t+1 in budgets:
                    lp=log_posteriors(tpl,obs,mask,noise)
                    key=str(t+1)
                    totals[name]['correct'][key]+=(lp.argmax(1)==targets).sum().item()
                    totals[name]['logp'][key]+=lp[idx,targets].sum().item()
        done+=B
    return {name:{'accuracy':{k:round(v/n,4) for k,v in totals[name]['correct'].items()},
                  'mean_log_p_true':{k:round(v/n,4) for k,v in totals[name]['logp'].items()},
                  'first_choices':totals[name]['first_choices']}
            for name in policies}


def run(args):
    seed_all(SEED)
    m=load_model()
    train_ids,test_ids=split_templates(m['embed'])
    test=EpisodeGenerator(m['embed'],test_ids,args.noise,seed=120)
    experiments=[('pretrained_tuned',True,True),('pretrained_frozen',True,False),('random_tuned',False,True)]
    models={}; traces={}
    for i,(name,pretrained,tune) in enumerate(experiments):
        print('Training',name,'...',flush=True)
        train=EpisodeGenerator(m['embed'],train_ids,args.noise,seed=110)
        model, trace=train_policy(m,train,args.noise,args.steps,pretrained,tune,seed=400+i,batch_size=args.batch)
        models[name]=model
        traces[name]=trace
        print(' ', trace[-1],flush=True)
    scores=evaluate(models,test,args.noise,n=args.trials)
    print('HOLDOUT',json.dumps({k:v['accuracy'] for k,v in scores.items()},indent=2),flush=True)
    result={'setup':{'seed':SEED,'train_word_ids':train_ids.tolist(),'heldout_word_ids':test_ids.tolist(),
                     'train_word_count':len(train_ids),'heldout_word_count':len(test_ids),
                     'candidates_per_trial':NC,'noise_sd':args.noise,'train_steps':args.steps,
                     'train_batch':args.batch,'test_trials':args.trials,
                     'training':'all-action counterfactual reward from true hidden candidate; no variance imitation',
                     'evaluation':'disjoint word embeddings, no target information in policy inputs',
                     'pretrained_blocks':'copy QKV/output/MLP weights from existing index.html, then tune/freeze as stated',
                     'architecture_change':'bidirectional attention over 24 candidate coordinate tokens, using original RoPE and four blocks',
                     'decoder':'same Gaussian Bayesian classifier for every arm; it updates from observed coordinates only',
                     'limitations':'shortlists consist of a finite 70-word vocabulary; ideal decoder; no genuine external world'},
            'train_trace':traces,'scores':scores}
    dest=HERE/'results_learned_policy.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    # Only the trained policy weights, no copy of large original model weights.
    torch.save(models['pretrained_tuned'].state_dict(), HERE/'learned_policy.pt')
    print('Wrote',dest,flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--batch',type=int,default=96)
    ap.add_argument('--noise',type=float,default=.9)
    ap.add_argument('--trials',type=int,default=480)
    run(ap.parse_args())