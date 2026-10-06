# Qwen user-side latent reasoning

User-approved scope: reuse Qwen3-Embedding-8B tokenizer assets; use
Qwen/Qwen3-4B-Instruct-2507 with LoRA and structural projection; execute two
continuous latent steps with a separate persistent memory; commit to SIR-CDR.

The existing Text v4 implementation and checkpoints remain a baseline. A new
LLM config and runner build TextSIRCDR and replace its reasoner through the
existing `(context, cd_signal, target_private)` interface. Three learned
structural tokens and a learned latent seed condition the Qwen3 base transformer.
This first version uses projected structural inputs, not raw history text.
This is an AGCLR-inspired recommendation adaptation, not the original CoT
curriculum or a claim of measured recommendation gains.

At each step obtain Qwen's last hidden state h; compute read/forget/write gates;
set h'=(1-f)*h+r*memory and memory=LN(memory+w*h'). Append h' as the next
continuous input embedding and execute Qwen again. Memory persists across the
two steps, resets for each batch, and is never detached. Cache is disabled and
the short prefix is recomputed to keep the autograd path unambiguous. Project
the final state to the existing shared/private heads. Default CPF rounds=1.

Use actual Qwen3Model and PEFT FEATURE_EXTRACTION LoRA on q_proj/v_proj.
Train only LoRA and recommendation/projection/memory modules. Do not run the
offline embedding model in the training process. First implementation is BF16
LoRA on CUDA (float32 CPU tests), not 4-bit quantization. Microbatch=1 and
gradient accumulation reduce memory; no unverified 24GB guarantee.

Persist compact checkpoints (all non-backbone state plus trainable adapters),
optimizer/scheduler/RNG state for epoch-boundary resume. Pin HF commit SHA and
library/code/asset hashes; include label smoothing in this new identity. Refuse
incompatible checkpoints and changed test evaluation identities. Use isolated
artifacts/llm_cdr_v1 output. Baseline ranking/decoder stay unchanged.

Acceptance: real tiny random Qwen3+PEFT CPU tests prove two LM passes, memory
persistence/reset, recurrent gradients, frozen base parameters, projection and
adapter training, ranking label independence, compact checkpoint round-trip,
epoch resume, final accumulation batch, and immutable test identity. Real 4B
GPU fit/quality are remote acceptance tasks, not inferred from CPU tests.
