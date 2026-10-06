"""Differentiable Qwen passes with an independent, per-example concept memory.

No text generation API and no pretrained embedding model is used here. Frozen
base weights still participate in autograd so projections and earlier thoughts
receive gradients. Recompute the short prefix instead of detaching a KV cache.
"""
import torch
from torch import nn


class PersistentMemory(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_norm = nn.LayerNorm(width)
        self.read = nn.Linear(width, width)
        self.forget = nn.Linear(width, width)
        self.write = nn.Linear(width, width)
        self.memory_norm = nn.LayerNorm(width)
        for layer, bias in ((self.read, 0.0), (self.forget, -1.0), (self.write, -1.0)):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, bias)

    def forward(self, hidden, memory):
        normalized = self.input_norm(hidden)
        read = torch.sigmoid(self.read(normalized))
        forget = torch.sigmoid(self.forget(normalized))
        write = torch.sigmoid(self.write(normalized))
        thought = (1 - forget) * hidden + read * memory
        return thought, self.memory_norm(memory + write * thought)


class QwenLatentReasoner(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.reasoning_steps = config.reasoning_steps
        width = backbone.config.hidden_size
        hidden = config.hidden_dim
        self.project = nn.Linear(hidden, width)
        self.token_norm = nn.LayerNorm(width)
        self.token_types = nn.Parameter(torch.empty(3, width))
        self.seed = nn.Parameter(torch.empty(1, width))
        nn.init.normal_(self.token_types, std=0.02)
        nn.init.normal_(self.seed, std=0.02)
        self.memory = PersistentMemory(width)
        self.to_state = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.shared_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.private_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.Tanh())

    @classmethod
    def from_pretrained(cls, config):
        # Lazy imports: the original Text v4 path has no additional dependencies.
        from transformers import AutoModel
        from peft import LoraConfig, TaskType, get_peft_model
        config.require_pinned_model()
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[config.llm_dtype]
        base = AutoModel.from_pretrained(config.llm_model_name, revision=config.llm_revision,
            torch_dtype=dtype, local_files_only=config.llm_local_files_only,
            trust_remote_code=False, attn_implementation="sdpa")
        if base.config.model_type != "qwen3":
            raise ValueError("LLM v1 supports the Qwen3 dense text backbone only")
        if getattr(base.config, "_commit_hash", None) != config.llm_revision:
            raise ValueError("Loaded backbone commit does not match pinned revision")
        base.config.use_cache = False
        base = get_peft_model(base, LoraConfig(r=config.lora_rank,
            lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj"], task_type=TaskType.FEATURE_EXTRACTION))
        if config.llm_gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(base, config)

    def forward(self, context, cd_signal, target_sp_signal):
        structural = torch.stack([context, cd_signal, target_sp_signal], dim=1)
        prefix = self.token_norm(self.project(structural) + self.token_types)
        tokens = torch.cat([prefix, self.seed.expand(len(context), -1, -1)], dim=1)
        dtype = self.backbone.get_input_embeddings().weight.dtype
        tokens = tokens.to(dtype=dtype)
        memory = prefix.new_zeros(len(context), prefix.shape[-1])
        states = []
        for step in range(self.reasoning_steps):
            mask = torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.long)
            output = self.backbone(inputs_embeds=tokens, attention_mask=mask,
                                   use_cache=False, return_dict=True)
            hidden = output.last_hidden_state[:, -1].to(dtype=prefix.dtype)
            thought, memory = self.memory(hidden, memory)
            states.append(self.to_state(thought))
            if step + 1 < self.reasoning_steps:
                tokens = torch.cat([tokens, thought[:, None].to(dtype=dtype)], dim=1)
        final = states[-1]
        shared = self.shared_head(torch.cat([final, cd_signal], dim=-1))
        private = self.private_head(torch.cat([final, target_sp_signal], dim=-1))
        return shared, private, torch.stack(states, dim=1)
