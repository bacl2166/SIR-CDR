from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from cdr_framework.codebook import ItemTokenIndex, RecommendationHit, SemanticCodebook, beam_search
from cdr_framework.data import ForwardOutput, GraphBatch, InteractionBatch, ItemFeatures
from cdr_framework.losses import (
    anti_collapse_variance_loss,
    orthogonality_loss,
    shared_alignment_loss,
    source_private_separation_loss,
)
from cdr_framework.modules import (
    AutoregressiveSemanticDecoder,
    CrossDomainStructuralInjector,
    DualStructuralFusionPrefix,
    MultimodalSemanticEncoder,
    PrototypeDisentangler,
    SharedSpecificGraphEncoder,
    SpecificDomainStructuralInjector,
    UserDisentangler,
    UserImplicitReasoner,
)
from cdr_framework.ops import sequence_mean


@dataclass(frozen=True)
class EncodedState:
    semantic_tokens: torch.Tensor
    cd_graph: torch.Tensor
    source_graph: torch.Tensor
    target_graph: torch.Tensor
    shared_item_tokens: torch.Tensor
    source_specific_tokens: torch.Tensor
    target_specific_tokens: torch.Tensor
    source_user_specific: torch.Tensor
    target_user_specific: torch.Tensor
    shared_latent: torch.Tensor
    private_latent: torch.Tensor
    cd_signal: torch.Tensor
    source_sp_signal: torch.Tensor
    target_sp_signal: torch.Tensor
    prefix: torch.Tensor
    reasoning_states: torch.Tensor
    vq_loss: torch.Tensor
    target_tokens_by_item: torch.Tensor


class DualStructuralInjectionRecommender(nn.Module):
    def __init__(
        self,
        encoder: MultimodalSemanticEncoder,
        graph_encoder: SharedSpecificGraphEncoder,
        prototype_disentangler: PrototypeDisentangler,
        user_disentangler: UserDisentangler,
        cd_injector: CrossDomainStructuralInjector,
        sp_injector: SpecificDomainStructuralInjector,
        reasoner: UserImplicitReasoner,
        codebook: SemanticCodebook,
        prefix: DualStructuralFusionPrefix,
        decoder: AutoregressiveSemanticDecoder,
        hidden_dim: int,
        token_length: int,
        prefix_length: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.graph_encoder = graph_encoder
        self.prototype_disentangler = prototype_disentangler
        self.user_disentangler = user_disentangler
        self.cd_injector = cd_injector
        self.sp_injector = sp_injector
        self.reasoner = reasoner
        self.codebook = codebook
        self.prefix = prefix
        self.decoder = decoder
        self.hidden_dim = hidden_dim
        self.token_length = token_length
        self.prefix_length = prefix_length
        self.item_token_index = ItemTokenIndex()

    @classmethod
    def random_init(
        cls,
        num_items: int,
        txt_dim: int,
        img_dim: int,
        attr_dim: int,
        hidden_dim: int,
        codebook_size: int,
        token_length: int,
        seed: int = 0,
        prefix_length: int = 4,
        reasoning_steps: int = 3,
    ) -> "DualStructuralInjectionRecommender":
        del num_items
        torch.manual_seed(seed)
        codebook = SemanticCodebook.random_init(codebook_size, hidden_dim, seed + 1)
        return cls(
            encoder=MultimodalSemanticEncoder(hidden_dim, txt_dim, img_dim, attr_dim, hidden_dim),
            graph_encoder=SharedSpecificGraphEncoder(hidden_dim),
            prototype_disentangler=PrototypeDisentangler(hidden_dim),
            user_disentangler=UserDisentangler(hidden_dim),
            cd_injector=CrossDomainStructuralInjector(hidden_dim),
            sp_injector=SpecificDomainStructuralInjector(hidden_dim),
            reasoner=UserImplicitReasoner(hidden_dim, reasoning_steps),
            codebook=codebook,
            prefix=DualStructuralFusionPrefix(hidden_dim, prefix_length),
            decoder=AutoregressiveSemanticDecoder(codebook_size, hidden_dim, prefix_length),
            hidden_dim=hidden_dim,
            token_length=token_length,
            prefix_length=prefix_length,
        )

    def forward(self, batch: InteractionBatch, item_features: ItemFeatures, graph: GraphBatch) -> ForwardOutput:
        state = self.encode_state(batch, item_features, graph)
        target_tokens = state.target_tokens_by_item[batch.positive_target_items.long()]
        decoder_input = self._decoder_input(target_tokens)
        decoder_target = self._decoder_target(target_tokens)
        logits = self.decoder(state.prefix, decoder_input)
        losses = self._compute_losses(batch, item_features, state, logits, decoder_target)
        return ForwardOutput(prefix=state.prefix, target_tokens=target_tokens, logits=logits, losses=losses)

    def encode_state(self, batch: InteractionBatch, item_features: ItemFeatures, graph: GraphBatch) -> EncodedState:
        semantic = self.encoder(
            item_features.id_vectors,
            item_features.text_vectors,
            item_features.image_vectors,
            item_features.attr_vectors,
        )
        cd_graph, source_graph, target_graph = self.graph_encoder(semantic, graph)
        proto_shared, proto_source_sp, proto_target_sp = self.prototype_disentangler(semantic, cd_graph)
        shared_item_tokens = torch.tanh(semantic + cd_graph + proto_shared)
        source_specific_tokens = torch.tanh(semantic + source_graph + proto_source_sp)
        target_specific_tokens = torch.tanh(semantic + target_graph + proto_target_sp)
        target_fused_tokens = torch.tanh(shared_item_tokens + target_specific_tokens)
        target_tokens_by_item, _, vq_loss = self.codebook.quantize(target_fused_tokens, self.token_length)

        context = sequence_mean(semantic, torch.cat([batch.source_sequences, batch.target_sequences], dim=1).long())
        user_shared, source_user_specific, target_user_specific = self.user_disentangler(context)
        cd_signal = self.cd_injector(
            shared_item_tokens,
            cd_graph,
            batch.source_sequences,
            batch.target_sequences,
            user_shared=user_shared,
        )
        source_sp_signal, target_sp_signal = self.sp_injector(
            source_specific_tokens,
            source_graph,
            target_specific_tokens,
            target_graph,
            batch.source_sequences,
            batch.target_sequences,
        )
        shared_latent, private_latent, reasoning_states = self.reasoner(context, cd_signal, target_sp_signal)
        codebook_summary = self.codebook.pooled(shared_latent + private_latent)
        prefix = self.prefix(shared_latent, private_latent, cd_signal, target_sp_signal, codebook_summary)
        return EncodedState(
            semantic_tokens=semantic,
            cd_graph=cd_graph,
            source_graph=source_graph,
            target_graph=target_graph,
            shared_item_tokens=shared_item_tokens,
            source_specific_tokens=source_specific_tokens,
            target_specific_tokens=target_specific_tokens,
            source_user_specific=source_user_specific,
            target_user_specific=target_user_specific,
            shared_latent=shared_latent,
            private_latent=private_latent,
            cd_signal=cd_signal,
            source_sp_signal=source_sp_signal,
            target_sp_signal=target_sp_signal,
            prefix=prefix,
            reasoning_states=reasoning_states,
            vq_loss=vq_loss,
            target_tokens_by_item=target_tokens_by_item,
        )

    def recommend(
        self,
        batch: InteractionBatch,
        item_features: ItemFeatures,
        graph: GraphBatch,
        top_k: int,
        beam_size: int = 5,
        max_length: int | None = None,
    ) -> list[list[RecommendationHit]]:
        state = self.encode_state(batch, item_features, graph)
        self._rebuild_item_index(item_features, state)
        max_len = max_length or self.token_length
        all_recs: list[list[RecommendationHit]] = []
        for row in range(batch.batch_size):
            prefix = state.prefix[row : row + 1]

            def step(tokens: list[int]) -> dict[int, float]:
                log_probs = self.decoder.next_log_probs(prefix, tokens)
                values, ids = torch.topk(log_probs, k=min(beam_size, log_probs.shape[0]))
                return {int(ids[idx].item()): float(values[idx].item()) for idx in range(ids.shape[0])}

            beams = beam_search(
                step_log_probs=step,
                bos_id=self.decoder.bos_id,
                eos_id=self.decoder.eos_id,
                beam_size=beam_size,
                max_length=max_len,
            )
            hits: list[RecommendationHit] = []
            query = state.prefix[row].mean(dim=0)
            for result in beams:
                hits.extend(self.item_token_index.lookup_ranked(result.tokens, result.score, query, top_k))
            if not hits:
                hits.extend(self.item_token_index.nearest_ranked(query, top_k))
            hits.sort(key=lambda hit: hit.score, reverse=True)
            all_recs.append(hits[:top_k])
        return all_recs

    def _compute_losses(
        self,
        batch: InteractionBatch,
        item_features: ItemFeatures,
        state: EncodedState,
        logits: torch.Tensor,
        decoder_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gen = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), decoder_target.reshape(-1))
        final_hidden = logits[:, -1, : self.hidden_dim]
        pos_repr = state.target_specific_tokens[batch.positive_target_items.long()]
        idx = 1.0 - F.cosine_similarity(final_hidden, pos_repr, dim=-1).mean()
        source_mask = torch.tensor([domain == "S" for domain in item_features.domains], device=logits.device)
        target_mask = torch.tensor([domain == "T" for domain in item_features.domains], device=logits.device)
        sh = shared_alignment_loss(state.shared_item_tokens[source_mask], state.shared_item_tokens[target_mask])
        orth = orthogonality_loss(state.shared_item_tokens, state.target_specific_tokens)
        sep = source_private_separation_loss(state.private_latent, state.source_sp_signal)
        cpf = anti_collapse_variance_loss(state.reasoning_states)
        losses = {
            "gen": gen,
            "idx": idx,
            "vq": state.vq_loss,
            "shared_alignment": sh,
            "orth": orth,
            "sp_sep": sep,
            "cpf": cpf,
        }
        losses["total"] = gen + 0.1 * idx + 0.1 * state.vq_loss + 0.05 * sh + 0.05 * orth + 0.05 * sep + 0.05 * cpf
        return losses

    def _decoder_input(self, target_tokens: torch.Tensor) -> torch.Tensor:
        bos = torch.full((target_tokens.shape[0], 1), self.decoder.bos_id, dtype=torch.long, device=target_tokens.device)
        return torch.cat([bos, target_tokens.long()], dim=1)

    def _decoder_target(self, target_tokens: torch.Tensor) -> torch.Tensor:
        eos = torch.full((target_tokens.shape[0], 1), self.decoder.eos_id, dtype=torch.long, device=target_tokens.device)
        return torch.cat([target_tokens.long(), eos], dim=1)

    def _rebuild_item_index(self, item_features: ItemFeatures, state: EncodedState) -> None:
        self.item_token_index.clear()
        for item_id in item_features.target_item_ids().tolist():
            self.item_token_index.add(
                int(item_id),
                state.target_tokens_by_item[int(item_id)].tolist(),
                state.target_specific_tokens[int(item_id)],
            )
