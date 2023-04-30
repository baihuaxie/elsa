"""GPT-2 model. Based on Huggingface code base.
"""
import math
import torch
import torch.nn as nn

from .attention.sdp import ScaledDotProductAttentionForGPT


class GPTConfig:
    """Configuration for GPT-2 model with vanilla attention.
    """
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_embed=768,
        n_head=8,
        n_layer=12,
        hidden_dropout_prob=0.0,
        layer_norm_eps=1e-5,
        layer_norm_bias=True,
        bias=True,
        flash=True,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_layer = n_layer
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_bias = layer_norm_bias
        self.bias = bias
        self.flash = flash


class GPTLayerNorm(nn.Module):
    """LayerNOrm with optional bias.
    """
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embed))
        self.bias = nn.Parameter(torch.zeros(config.n_embed)) if config.layer_norm_bias else None
        self.layer_norm_eps = config.layer_norm_eps

    def forward(self, x):
        return nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, self.layer_norm_eps
        )


class GPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPTBlock(nn.Module):
    def __init__(self, config, attention: nn.Module):
        super().__init__()
        self.ln_1 = GPTLayerNorm(config)
        self.attn = attention
        self.ln_2 = GPTLayerNorm(config)
        self.mlp = GPTMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLMHeadModel(nn.Module):
    _keys_to_ignore_on_load_unexpected = ["attn.masked_bias"]
    def __init__(self, config, attention: nn.Module):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            dropout = nn.Dropout(config.hidden_dropout_prob),
            h = nn.ModuleList([
                GPTBlock(config, attention) for _ in range(config.n_layer)
            ]),
            ln_f = GPTLayerNorm(config)
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.block_size = config.block_size

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # scaled init per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                # note: don't use p.weight...; Paramter object has no `weight` member.
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)


    @classmethod
    def from_pretrained(cls, model_type="gpt2", attention_type=None):
        """Load HuggingFace pretrained model weights.
        """
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        # ignore `attn.bias`, this is just the causal mask buffer
        # ignore `attn.masked_bias`, this is present in HF's code base but never used
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith("attn.bias")]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith("attn.masked_bias")]

        config = GPTConfig(**config_args)
        if attention_type is None:
            attention = ScaledDotProductAttentionForGPT
        model = GPTLMHeadModel(
            config=config,
            attention=attention,
        )
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith("attn.bias")]

        assert set(sd_keys) == set(sd_hf_keys)
        # in HuggingFace's pretrained checkpoint (as in OpenAI's original checkpoints)
        # they use Conv1D instead of nn.Linear() for the projection layers
        # the weights will need to be transposed -> see notes for details on this one
        transposed_keys = [
            "attn.c_attn.weight", "attn.c_proj.weight","mlp.c_fc.weight", "mlp.c_proj.weight"
        ]
        for k in sd_hf_keys:
            # special handling for Conv1D to Linear weight copying
            # pylint: disable=unsubscriptable-object
            if any(k.endswith(w) for w in transposed_keys):
                assert sd[k].shape == sd_hf[k].shape[::-1]
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


    def forward(self, input_ids, labels):
        b, t, device = *input_ids.shape, input_ids.device
        assert t <= self.block_size

        # absolute position bias
        # shape = (1, t)
        position_ids = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0)
        position_bias = self.transformer.wpe(position_ids)
        hidden_states = self.transformer.wte(input_ids)
        hidden_states = self.transformer.dropout(hidden_states + position_bias)

        for block in self.transformer.h:
            hidden_states = block(hidden_states)
        hidden_states = self.transformer.ln_f(hidden_states)

        logits = self.lm_head(hidden_states)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        else:
            loss = None

        return loss, logits





