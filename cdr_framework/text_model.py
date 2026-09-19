from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from cdr_framework.modules import (
    UserImplicitReasoner,
    AutoregressiveSemanticDecoder,
    DualStructuralFusionPrefix,
    TextPrototypeDisentangler,
    CodebookSummaryPool,
    CrossDomainStructuralInjector,
    SpecificDomainStructuralInjector,
    GatedSignalFusion,
)
from cdr_framework.modules_cpf import ContextPredictionFeedback
from cdr_framework.losses import (
    orthogonality_loss,
    anti_collapse_variance_loss,
    shared_alignment_loss,
    source_private_separation_loss,
)
from cdr_framework.text_graph import SparseDualGraphEncoder
from cdr_framework.catalog_generation import CatalogTrie, constrained_generate


class TextSIRCDR(nn.Module):
    """Text-derived content, sparse structural injection and recurrent latent feedback."""

    def __init__(self, catalog, graphs, config):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.length, self.size = catalog.codebooks.shape[:2]
        self.register_buffer("content", catalog.item_latents.float(), persistent=False)
        offsets = torch.arange(self.length) * self.size
        tokens = catalog.semantic_ids.long() + offsets
        tokens[0] = 0
        self.register_buffer("tokens", tokens, persistent=False)
        self.register_buffer("target_ids", catalog.target_item_ids.long(), persistent=False)
        self.register_buffer("centroids", catalog.codebooks.flatten(0, 1).float(), persistent=False)
        for name, value in graphs.items():
            self.register_buffer(f"graph_{name}", value, persistent=False)
        index = torch.full((len(self.content),), -1, dtype=torch.long)
        index[self.target_ids] = torch.arange(len(self.target_ids))
        self.register_buffer("target_index", index, persistent=False)
        self.content_proj = nn.Linear(self.content.shape[1], h)
        self.code_proj = nn.Linear(self.centroids.shape[1], h)
        self.item_norm = nn.LayerNorm(h)
        self.graph_encoder = SparseDualGraphEncoder(h, layers=config.graph_layers)
        self.prototype = TextPrototypeDisentangler(h)
        self.cd_injector = CrossDomainStructuralInjector(h)
        self.sp_injector = SpecificDomainStructuralInjector(h)
        self.source_private_fusion = GatedSignalFusion(h)
        self.target_private_fusion = GatedSignalFusion(h)
        self.codebook_summary = CodebookSummaryPool(h)
        self.structural_gate = nn.Linear(3 * h, h)
        self.source_gru, self.target_gru = nn.GRU(h, h, batch_first=True), nn.GRU(h, h, batch_first=True)
        self.attention = nn.Linear(h, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.shared_head, self.source_head, self.target_head = [nn.Linear(h, h) for _ in range(3)]
        self.transfer_gate = nn.Linear(2 * h, h)
        self.reasoner = UserImplicitReasoner(h, config.reasoning_steps)
        self.cpf = ContextPredictionFeedback(h, h)
        self.feedback_gate, self.feedback_proj = nn.Linear(2 * h, h), nn.Linear(h, h)
        self.query_head = nn.Sequential(nn.Linear(3 * h, h), nn.LayerNorm(h))
        self.prefix_head = DualStructuralFusionPrefix(h, config.prefix_length)
        self.decoder = AutoregressiveSemanticDecoder(self.length * self.size, h, config.prefix_length)
        self.trie = CatalogTrie(self.target_ids.cpu(), self.tokens.cpu())

    def item_states(self):
        values = self.content_proj(self.content)
        if self.config.semantic_enabled:
            code_vectors = self.code_proj(self.centroids)[self.tokens].mean(1)
            values = values + code_vectors
        values = self.item_norm(values)
        values = values * (torch.arange(len(values), device=values.device) != 0)[:, None]
        if self.config.graph_enabled:
            shared, source, target = self.graph_encoder(values, {
                name: getattr(self, f"graph_{name}") for name in ("shared", "source", "target")
            })
        else:
            shared = source = target = torch.zeros_like(values)
        proto = None
        if self.config.prototype_enabled:
            proto = self.prototype(values, shared, source, target)
        return {"base": values, "shared": shared, "source": source, "target": target, "proto": proto}

    def sequence(self, base, shared, specific, ids, lengths, gru):
        gate = torch.sigmoid(self.structural_gate(torch.cat([base[ids], shared[ids], specific[ids]], -1)))
        inputs = self.dropout(base[ids] + gate * shared[ids] + (1 - gate) * specific[ids])
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, final = gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=ids.shape[1])
        mask = torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, -torch.inf)
        pooled = (scores.softmax(1)[..., None] * encoded).sum(1)
        return 0.5 * (pooled + final[-1])

    def encode(self, batch, items=None):
        items = self.item_states() if items is None else items
        base, graph_shared, graph_source, graph_target = items["base"], items["shared"], items["source"], items["target"]
        proto = items["proto"]
        source = self.sequence(base, graph_shared, graph_source, batch.source_items, batch.source_lengths, self.source_gru)
        target = self.sequence(base, graph_shared, graph_target, batch.target_items, batch.target_lengths, self.target_gru)
        if not self.config.source_enabled:
            source = torch.zeros_like(source)
        source_shared, target_shared = torch.tanh(self.shared_head(source)), torch.tanh(self.shared_head(target))
        source_private_sequence = torch.tanh(self.source_head(source))
        target_private_sequence = torch.tanh(self.target_head(target))
        transfer = torch.sigmoid(self.transfer_gate(torch.cat([source, target], -1)))
        shared_signal = transfer * source_shared + (1 - transfer) * target_shared
        context = target + transfer * source
        if self.config.cd_injector_enabled:
            shared_tokens = proto["tokens_shared"] if proto is not None else base
            cd_signal = self.cd_injector(
                shared_tokens, graph_shared, batch.source_items, batch.target_items,
                batch.source_lengths, batch.target_lengths,
                user_shared=shared_signal,
            )
        else:
            cd_signal = shared_signal
        if self.config.sp_injector_enabled:
            source_tokens = proto["tokens_source"] if proto is not None else base
            target_tokens = proto["tokens_target"] if proto is not None else base
            source_sp_signal, target_sp_signal = self.sp_injector(
                source_tokens, graph_source, target_tokens, graph_target,
                batch.source_items, batch.target_items,
                batch.source_lengths, batch.target_lengths,
            )
            source_private = self.source_private_fusion(source_private_sequence, source_sp_signal)
            target_private = self.target_private_fusion(target_private_sequence, target_sp_signal)
        else:
            source_sp_signal, target_sp_signal = source_private_sequence, target_private_sequence
            source_private, target_private = source_private_sequence, target_private_sequence
        feedback_losses = []
        states = []
        rounds = self.config.feedback_steps if self.config.reasoning_enabled else 1
        for round_index in range(rounds):
            if self.config.reasoning_enabled:
                shared, private, reasoning = self.reasoner(context, cd_signal, target_private)
            else:
                shared, private = shared_signal, target_private
                reasoning = (shared + private)[:, None]
            prediction = self.cpf.predictor(torch.cat([shared + private, context], -1))
            feedback_losses.append(prediction)
            states.append(reasoning)
            if round_index + 1 < rounds:
                gate = torch.sigmoid(self.feedback_gate(torch.cat([context, prediction], -1)))
                context = context + gate * torch.tanh(self.feedback_proj(prediction))
        query = F.normalize(self.query_head(torch.cat([shared, private, target], -1)), dim=-1)
        if self.config.codebook_summary_enabled:
            summary = self.codebook_summary(shared + private, self.centroids)
        else:
            summary = prediction
        prefix = self.prefix_head(shared, private, cd_signal, target_private, summary)
        return dict(query=query, prefix=prefix, shared=shared, private=private,
                    source_private=source_private, source_shared=source_shared, target_shared=target_shared,
                    context=context, states=torch.cat(states, 1), predictions=feedback_losses,
                    source_sp_signal=source_sp_signal, target_sp_signal=target_sp_signal,
                    target_private=target_private, proto=proto)

    def forward(self, batch):
        items = self.item_states()
        state = self.encode(batch, items)
        labels = self.target_index[batch.positive_items]
        if (labels < 0).any():
            raise ValueError("Positive item outside target domain")
        base = items["base"]
        retrieval = state["query"] @ F.normalize(base[self.target_ids], dim=-1).T / self.config.retrieval_temperature
        retrieval_loss = F.cross_entropy(retrieval, labels)
        tokens = self.tokens[batch.positive_items]
        bos = torch.full((len(tokens), 1), self.decoder.bos_id, dtype=torch.long, device=tokens.device)
        expected = torch.cat([tokens, torch.full_like(bos, self.decoder.eos_id)], 1)
        logits = self.decoder(state["prefix"], torch.cat([bos, tokens], 1))
        generation = F.cross_entropy(logits.flatten(0, 1), expected.flatten(),
                                     label_smoothing=self.config.label_smoothing)
        # Stable text target, detached so the auxiliary objective cannot move its own target.
        positive = base[batch.positive_items].detach()
        cpf = torch.stack([1 - F.cosine_similarity(p, positive).mean() for p in state["predictions"]]).mean()
        cpf = cpf + 0.1 * (1 - F.cosine_similarity(state["shared"], state["context"]).mean())
        cpf = cpf + 0.1 * anti_collapse_variance_loss(state["states"])
        if self.config.contrastive_alignment:
            alignment = shared_alignment_loss(state["source_shared"], state["target_shared"])
        else:
            alignment = (1 - F.cosine_similarity(state["source_shared"], state["target_shared"])).mean()
        if not self.config.source_enabled:
            alignment = alignment * 0
        separation = orthogonality_loss(state["shared"], state["private"])
        if self.config.source_enabled:
            separation = separation + orthogonality_loss(state["private"], state["source_private"])
        lsep = state["shared"].new_tensor(0.0)
        if self.config.sp_injector_enabled and self.config.lsep_weight > 0:
            lsep = source_private_separation_loss(state["private"], state["source_private"])
        proto_orth = state["shared"].new_tensor(0.0)
        if self.config.prototype_enabled and self.config.proto_orth_weight > 0:
            proto = state["proto"]
            proto_orth = orthogonality_loss(proto["proto_shared"][1:], proto["proto_target"][1:])
        total = (self.config.generation_loss_weight * generation
                 + self.config.retrieval_loss_weight * retrieval_loss
                 + self.config.cpf_weight * cpf)
        total = total + self.config.alignment_weight * alignment + self.config.separation_weight * separation
        total = total + self.config.lsep_weight * lsep + self.config.proto_orth_weight * proto_orth
        return {"total": total, "generation": generation, "retrieval": retrieval_loss,
                "cpf": cpf, "alignment": alignment, "separation": separation,
                "lsep": lsep, "proto_orth": proto_orth}

    def sequence_scores(self, prefix, item_ids):
        tokens = self.tokens[item_ids]
        bos = torch.full((len(tokens), 1), self.decoder.bos_id, dtype=torch.long, device=tokens.device)
        targets = torch.cat([tokens, torch.full_like(bos, self.decoder.eos_id)], 1)
        logits = self.decoder(prefix, torch.cat([bos, tokens], 1))
        return F.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1).mean(1)

    @torch.no_grad()
    def rank(self, batch, seen, top_k, mode=None):
        mode = mode or self.config.inference_mode
        items = self.item_states()
        state = self.encode(batch, items)
        base = F.normalize(items["base"], dim=-1)
        if mode == "generate":
            return constrained_generate(self.decoder, state["prefix"], self.trie,
                beam_size=self.config.beam_size, top_k=top_k, seen_items=seen,
                query=state["query"], item_vectors=base)
        if mode not in {"retrieval", "hybrid", "exhaustive"}:
            raise ValueError("Unknown ranking mode")
        scores = state["query"] @ base[self.target_ids].T / self.config.retrieval_temperature
        for row, history in enumerate(seen):
            indices = self.target_index[torch.tensor(sorted(history), device=scores.device, dtype=torch.long)]
            scores[row, indices[indices >= 0]] = -torch.inf
        count = min(len(self.target_ids), max(top_k, self.config.rerank_candidates))
        if mode == "exhaustive":
            count = len(self.target_ids)
        values, indices = scores.topk(count, dim=1)
        candidates = self.target_ids[indices]
        if mode != "retrieval" and self.config.generation_score_weight > 0:
            batch_rows = torch.arange(len(scores), device=scores.device).repeat_interleave(count)
            flat_ids = candidates.flatten()
            gen = torch.empty(len(flat_ids), device=scores.device)
            for start in range(0, len(flat_ids), self.config.decode_chunk_size):
                stop = start + self.config.decode_chunk_size
                gen[start:stop] = self.sequence_scores(state["prefix"][batch_rows[start:stop]], flat_ids[start:stop])
            eligible = torch.isfinite(values)
            if getattr(self.config, "fusion_norm", "none") == "softmax":
                # Normalize both score families to probability scale before mixing
                # (retrieval cosine/temperature vs decoder log-probabilities have
                # incompatible magnitudes; softmax keeps the blend well-conditioned).
                values = torch.softmax(values, dim=-1)
                gen = torch.softmax(gen, dim=-1)
            values = (self.config.retrieval_score_weight * values
                      + self.config.generation_score_weight * gen.reshape_as(values))
            values = values.masked_fill(~eligible, -torch.inf)
        ranked_scores, order = values.topk(min(top_k, count), 1)
        ranked = candidates.gather(1, order).masked_fill(~torch.isfinite(ranked_scores), 0)
        return F.pad(ranked, (0, max(0, top_k - ranked.shape[1])))
