"""Target-domain prefix tree, masked softmax and label-free beam ranking."""
import math

import torch


class DomainTrie:
    def __init__(self, codes, items):
        self.children, self.leaves = {}, {}
        self.depth = codes.shape[1]
        for item in sorted(set(items)):
            if item <= 0:
                raise ValueError('Padding is not a candidate')
            code = tuple(codes[item].tolist())
            for k, token in enumerate(code):
                self.children.setdefault(code[:k], set()).add(token)
            self.leaves.setdefault(code, []).append(item)


def beam_rank(next_log_probs, trie, seen, *, beam=50, top_k=20):
    if beam < top_k or top_k < 1:
        raise ValueError('Beam must cover the requested ranking cutoff')
    states = [((), 0.)]
    for _ in range(trie.depth):
        logs = next_log_probs([prefix for prefix, _ in states]).detach().float().cpu()
        if not torch.isfinite(logs).all():
            raise RuntimeError('Non-finite generation scores')
        candidates = []
        for row, (prefix, score) in zip(logs, states):
            allowed = sorted(trie.children[prefix])
            # Paper Eq.12: normalize over valid next codes, never over the label.
            conditional = row[allowed].log_softmax(-1)
            for token, logp in zip(allowed, conditional.tolist()):
                candidates.append((prefix+(token,), score+logp))
        states = sorted(candidates, key=lambda p: (-p[1], p[0]))[:beam]
    ranked = []
    for code, score in states:
        items = trie.leaves[code]
        # Collision policy fixed before validation: equal mass, stable item-ID ties.
        ranked.extend((score-math.log(len(items)), item) for item in items if item not in seen)
    return [item for _, item in sorted(ranked, key=lambda p: (-p[0], p[1]))[:top_k]]


def ranking_metrics(rankings, positives):
    if len(rankings) != len(positives) or not positives:
        raise ValueError('One ranking per existing example required')
    result = {'examples': len(positives)}
    for k in (5, 10, 20):
        hits, ndcg = 0, 0.
        for ranked, positive in zip(rankings, positives):
            if len(ranked) != len(set(ranked)) or any(i <= 0 for i in ranked):
                raise ValueError('Ranking must contain unique real items')
            if positive in ranked[:k]:
                rank = ranked.index(positive)+1
                hits += 1
                ndcg += 1/math.log2(rank+1)
        result[f'HR@{k}'] = result[f'R@{k}'] = hits/len(positives)
        result[f'NDCG@{k}'] = result[f'N@{k}'] = ndcg/len(positives)
    return result
