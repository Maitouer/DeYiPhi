"""The H100 GRID-style Tiger graph, without legacy data or job dependencies."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerNorm, T5Stack


_T5_ATTENTION_FORWARD = T5Attention.forward


def _rope_tables(positions, dim, device, dtype):
    half = dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def _apply_rope(states, positions):
    """Rotate the last dimension of (batch, heads, length, head_dim) by absolute position."""
    cos, sin = _rope_tables(positions, states.shape[-1], states.device, states.dtype)
    cos, sin = cos[None, None], sin[None, None]
    half = states.shape[-1] // 2
    left, right = states[..., :half], states[..., half:]
    return torch.cat((left * cos - right * sin, left * sin + right * cos), dim=-1)


def _t5_sdpa_forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None,
                     past_key_value=None, layer_head_mask=None, query_length=None, use_cache=False,
                     output_attentions=False, cache_position=None):
    """``T5Attention.forward`` with RoPE instead of T5's relative-position bias.

    T5's bias is an *additive* mask that must be summed with the per-row padding mask and
    therefore broadcast to ``(batch, heads, tokens, tokens)``; every efficient kernel rejects
    that shape on the 2.4k-token full-history encoder, which is what forced the maths kernel
    (17 s/step, 38 GiB). With RoPE the only mask is a boolean padding/causal mask, which the
    mem-efficient kernel accepts at 0.11 GiB / 14.2 ms for 2435 tokens.

    Cross-attention keeps no positional encoding at all, exactly like T5's own
    ``has_relative_attention_bias=False`` cross-attention layers.
    """
    if (past_key_value is not None or output_attentions or layer_head_mask is not None
            or self.pruned_heads):
        return _T5_ATTENTION_FORWARD(self, hidden_states, mask, key_value_states, position_bias,
                                     past_key_value, layer_head_mask, query_length, use_cache,
                                     output_attentions, cache_position)
    batch_size, seq_length = hidden_states.shape[:2]
    shape = (batch_size, -1, self.n_heads, self.key_value_proj_dim)
    query_states = self.q(hidden_states).view(*shape).transpose(1, 2)
    current = key_value_states if key_value_states is not None else hidden_states
    key_states = self.k(current).view(*shape).transpose(1, 2)
    value_states = self.v(current).view(*shape).transpose(1, 2)
    if key_value_states is None:
        if cache_position is not None:
            positions = cache_position.to(torch.long)
        else:
            positions = torch.arange(key_states.shape[-2], device=query_states.device)
        query_states = _apply_rope(query_states, positions[-query_states.shape[-2]:])
        key_states = _apply_rope(key_states, positions[:key_states.shape[-2]])
    attn_mask = None if mask is None else mask > (torch.finfo(mask.dtype).min / 2)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0, scale=1.0)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    return self.o(attn_output), past_key_value, None


def enable_t5_sdpa():
    """Idempotently route T5 attention through RoPE + the mem-efficient SDPA kernel."""
    if T5Attention.forward is not _t5_sdpa_forward:
        T5Attention.forward = _t5_sdpa_forward
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def reset_parameters(module):
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()
    else:
        for child in module.children():
            reset_parameters(child)


class GridFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        d, h, p = config.d_model, config.d_ff, config.dropout_rate
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h),
                                 nn.Dropout(p), nn.ReLU(), nn.Linear(h, d), nn.Dropout(p))
        self.dropout = nn.Dropout(p)

    def forward(self, hidden_states):
        return hidden_states + self.dropout(self.mlp(self.layer_norm(hidden_states)))


class Tiger(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = int(cfg.codebook.width)
        options = dict(vocab_size=self.width, d_model=cfg.model.d_model, d_ff=cfg.model.d_ff,
                       d_kv=cfg.model.d_kv, num_heads=cfg.model.num_heads,
                       num_layers=cfg.model.num_layers, dropout_rate=cfg.model.dropout,
                       is_encoder_decoder=False)
        # transformers 4.53 has no native SDPA for T5, so "sdpa" installs our patched
        # attention (validated for numerical equivalence with the eager path).
        self.attention_implementation = str(cfg.model.get("attn_implementation", "eager"))
        if self.attention_implementation in ("rope", "sdpa"):
            enable_t5_sdpa()
        encoder_config = T5Config(**options)
        decoder_config = T5Config(**options, is_decoder=True)
        self.encoder = T5EncoderModel(encoder_config)
        self.decoder = T5Stack(decoder_config, nn.Embedding(self.width, cfg.model.d_model))
        del self.encoder.shared
        del self.encoder.encoder.embed_tokens
        del self.decoder.embed_tokens
        reset_parameters(self.encoder)
        reset_parameters(self.decoder)
        for stack, config in ((self.encoder.encoder, encoder_config), (self.decoder, decoder_config)):
            for block in stack.block:
                block.layer[-1] = GridFeedForward(config)
        self.sid_embeddings = nn.Embedding(4 * self.width, cfg.model.d_model)
        self.sep_token = nn.Parameter(torch.randn(1, cfg.model.d_model))
        self.bos_token = nn.Parameter(torch.randn(1, cfg.model.d_model))
        self.output_heads = nn.ModuleList(nn.Linear(cfg.model.d_model, self.width, bias=False) for _ in range(4))
        # phi (v6): the chain starts with **one** region token, so it needs its own embedding
        # rows and output head; every other variant keeps the four-level codebook untouched.
        self.phi_pg = cfg.get('phi', {}).get('variant', '') in ('v10_pg', 'v100')
        self.hierarchical = bool(cfg.get('phi', {}).get('hierarchical',False))
        self.g_levels = 2 if self.hierarchical else 1
        self.pg_user_prefix = bool(cfg.get('phi', {}).get('user_prefix', True))
        self.user_g_only = bool(cfg.get('phi', {}).get('user_g_only', False))
        self.empty_g_id = int(cfg.get('phi', {}).get('empty_g_id', -1))
        self.phi = cfg.get("phi", {}).get("variant", "") == "tree" or self.phi_pg
        self.phi_pred = cfg.get("phi", {}).get("variant", "") == "phi"
        # Analysis III's generation interface (05_analysis_design.md §16-17): "contextual"/"static"
        # keep a route in front of the SID chain, "none" drops it. The latter is the *memory only*
        # arm -- the encoder still receives the predictive tokens, but no route supervises the
        # target, so the chain is the native four-digit one.
        self.route_mode = str(cfg.get("phi", {}).get("route_mode", "contextual"))
        if self.phi_pred:
            # The predictive vocabulary (docs/v1_algorithm/03_phi.md §16.1): K interest tokens
            # in front of the native recent SIDs, and one route token in front of the generated
            # SID. ``vocabulary + 1`` classes per table: the extra class is the ``none`` sentinel a
            # masked slot or an unattributed target carries.
            classes = int(cfg.phi.vocabulary) + 1
            self.interest_embeddings = nn.Embedding(classes, cfg.model.d_model)
            self.route_embeddings = nn.Embedding(classes, cfg.model.d_model)
            reset_parameters(self.interest_embeddings)
            reset_parameters(self.route_embeddings)
            # The chain's first position *is* the route, so the existing chain machinery (one
            # prefix head followed by the SID heads) is reused unchanged.
            self.anchor_width = classes
            self.anchor_heads = nn.ModuleList([nn.Linear(cfg.model.d_model, classes, bias=False)])
            self.anchor_widths = [classes]
            self.g_levels = 1
            # The first decode step is restricted to the user's own tokens (docs 03_phi.md §17.1:
            # ``G_r = Unique{g_r1..g_rK}``), and an unattributed target carries the ``none`` class.
            self.user_g_only = True
            self.empty_g_id = classes - 1
        if self.phi:
            self.num_g1 = int(cfg.phi.num_g1)
            # The region token owns its classes *and* a "none" class (an item whose A leaf was
            # excluded from the tree), so target ids stay inside the head: 0..num_g1 = region,
            # num_g1 = none. The interest slot's second token is a *codebook digit* and reuses the
            # level-0 rows of ``sid_embeddings`` -- no second anchor table exists any more.
            self.anchor_width = self.num_g1 + (0 if self.phi_pg else 1)
            self.anchor_embeddings = nn.Embedding(self.anchor_width, cfg.model.d_model)
            reset_parameters(self.anchor_embeddings)
            self.anchor_heads = nn.ModuleList(
                [nn.Linear(cfg.model.d_model, self.anchor_width, bias=False)])
            self.anchor_widths = [self.anchor_width]
            if self.hierarchical:
                self.anchor_widths.append(int(cfg.phi.num_g2))
                self.anchor_heads.append(nn.Linear(cfg.model.d_model,int(cfg.phi.num_g2),bias=False))

        self.phi_v8 = cfg.get('phi', {}).get('variant', '') in ('v8', 'v9', 'v9_fast', 'v10_pg', 'v100')
        self.deyi_v8 = cfg.get('deyi', {}).get('variant', '') in ('v8', 'v9', 'v100')
        self.v8_teacher_root = (cfg.phi.teacher_root if self.phi_v8 else
            str(Path(cfg.deyi.root) / cfg.task / 'k4') if self.deyi_v8 else None)
        self.phi_v7 = cfg.get('phi', {}).get('variant', '') == 'v7'
        if self.phi_v7:
            import numpy as np
            payload = torch.load(cfg.phi.tree, map_location='cpu', weights_only=False)
            self.interest_embeddings = nn.Embedding(int(cfg.phi.leaves), cfg.model.d_model)
            self.interest_channel = nn.Embedding(3, cfg.model.d_model)
            self.interest_slot = nn.Embedding(4, cfg.model.d_model)
            with torch.no_grad():
                codes = np.load(Path(cfg.prepared_dir) / 'codes.npy', mmap_mode='r')
                native = torch.as_tensor(np.array(codes[payload['reference_rows']]), dtype=torch.long)
                reference = self.embed_sid(native).mean(1)
                value = payload['leaf_profile'].float() @ reference
                radius = self.sid_embeddings.weight.norm(dim=-1).mean()
                self.interest_embeddings.weight.copy_(F.normalize(value, dim=-1) * radius)
                nn.init.normal_(self.interest_channel.weight, std=.02)
                nn.init.normal_(self.interest_slot.weight, std=.02)
        self.prefix_sampling = float(cfg.get("train", {}).get("prefix_sampling", 0.0))
        deyi = cfg.get("deyi", {})
        self.deyi_enabled = bool(deyi.get("enabled", False))
        if self.deyi_enabled:
            state_dim = int(deyi.get("state_dim", 1024))
            slots = int(deyi.get("num_interests", 0))
            if slots < 1:
                raise ValueError("DeYi requires a positive number of interest slots")
            self.continuous_state_norm = nn.Identity() if self.deyi_v8 else nn.LayerNorm(state_dim)
            self.continuous_state_projection = nn.Linear(state_dim, cfg.model.d_model, bias=False)
            self.continuous_state_slots = nn.Embedding(slots, cfg.model.d_model)
        self.dummy_memory = bool(cfg.model.get("dummy_memory", False))
        self.dummy_slots = int(cfg.model.get("dummy_slots", 1))
        if self.dummy_memory:
            # ``dummy_slots`` learnable tokens shared by every user (06_analysis_baseline.md §1).
            # They carry no user information and are optimised only by the ordinary next-item
            # objective, so they isolate "additional trainable context" from predictive memory.
            self.dummy_memory_token = nn.Parameter(torch.zeros(self.dummy_slots, cfg.model.d_model))
            _radius = float(self.sid_embeddings.weight.detach().norm(dim=-1).mean())
            with torch.no_grad():
                self.dummy_memory_token.copy_(
                    torch.nn.functional.normalize(
                        torch.randn(self.dummy_slots, cfg.model.d_model), dim=-1) * _radius)
        # The Phi residual / joint variants were retired with the phi-tree v6 clean-up
        # (docs/new/phi_v6_final.md §8): the Tiger model keeps the native, DeYi-state and
        # phi paths only. ``src/model/phi`` is gone with them.
        self.radius_report = self._align_continuous_radii()
        if self.phi_v8:
            from src.model.deyi.prefix_v8 import calibrated_projection
            payload = torch.load(cfg.phi.tree, map_location='cpu', weights_only=False)
            if self.hierarchical:
                self.register_buffer('fine_to_coarse',payload['fine_to_coarse'].long())
            radius = float(self.sid_embeddings.weight.detach().norm(dim=-1).mean())
            projection, self.radius_report = calibrated_projection(self.v8_teacher_root, cfg.model.d_model, radius)
            self.interest_embeddings = nn.Embedding(len(payload['leaf_prototypes']), cfg.model.d_model)
            self.interest_slot = nn.Embedding(4, cfg.model.d_model)
            with torch.no_grad():
                self.interest_embeddings.weight.copy_(payload['leaf_prototypes'].float() @ projection.T)
                self.interest_slot.weight.zero_()
            # Only independent trainable rows remain; no prototype/projection in forward.
            assert self.interest_embeddings.weight.requires_grad
            if self.phi_pg:
                # A literal shared G vocabulary in user/history/decoder token sequences.
                # No residual addition to native SID embeddings, no auxiliary objective.
                self.anchor_embeddings = self.interest_embeddings
                with torch.no_grad():
                    if self.hierarchical:
                        self.anchor_heads[0].weight.copy_(self.interest_embeddings.weight[:self.num_g1])
                        self.anchor_heads[1].weight.copy_(self.interest_embeddings.weight[self.num_g1:])
                    else:
                        self.anchor_heads[0].weight.copy_(self.interest_embeddings.weight)


    def anchor_embeddings_of(self, anchor_ids):
        """Region ids -> embedding rows (v6 has exactly one anchor level)."""
        ids=anchor_ids.long().clamp_min(0)
        if self.hierarchical:
            offsets=ids.new_tensor([0,self.num_g1])[:ids.shape[-1]]
            ids=ids+offsets
        return self.anchor_embeddings(ids)

    def chain_head(self, depth):
        """Output head of chain position ``depth`` (the region, then the four codebook levels)."""
        if not (self.phi or self.phi_pred):
            return self.output_heads[depth]
        return self.anchor_heads[depth] if depth < self.g_levels else self.output_heads[depth - self.g_levels]

    def _align_continuous_radii(self):
        """Match the continuous prefix to the SID-token radius.

        DeYi exports raw states with radius ``sqrt(state_dim)``, so the prefix projection is an
        orthogonal matrix scaled so a
        typical state lands on the mean SID-token radius, the prefix LayerNorm starts as
        a no-op and the slot embeddings start at zero.
        """
        if not self.deyi_enabled:
            return {}
        if self.deyi_v8:
            from src.model.deyi.prefix_v8 import calibrated_projection
            radius = float(self.sid_embeddings.weight.detach().norm(dim=-1).mean())
            value, report = calibrated_projection(self.v8_teacher_root,
                self.continuous_state_projection.out_features, radius)
            with torch.no_grad():
                self.continuous_state_projection.weight.copy_(value)
                self.continuous_state_slots.weight.zero_()
            return report
        # 统一 target：4 个 SID 位置共用一个平均半径。
        sid_weight = self.sid_embeddings.weight.detach().float()
        per_position = sid_weight.view(4, self.width, -1).norm(dim=-1).mean(dim=1)
        target = float(per_position.mean())
        prefix = self.continuous_state_projection.weight
        hidden, state_dim = int(prefix.shape[0]), int(prefix.shape[1])
        with torch.no_grad():
            torch.nn.init.orthogonal_(prefix)
            prefix.mul_(target / math.sqrt(min(state_dim, hidden)))
            self.continuous_state_norm.weight.fill_(1.0)
            self.continuous_state_norm.bias.zero_()
            self.continuous_state_slots.weight.zero_()
        report = {"radius_alignment": "orthogonal_to_sid_token_radius_v1",
                  "target_radius": target, "state_dim": state_dim,
                  "hidden_size": hidden,
                  "prefix_scale": target / math.sqrt(min(state_dim, hidden)),
                  "sid_radius_by_position": [float(value) for value in per_position]}
        print("[radius] %s" % report, flush=True)
        return report

    def embed_sid(self, digits):
        offsets = torch.arange(digits.shape[-1], device=digits.device) * self.width
        return self.sid_embeddings(digits.long() + offsets)

    def encode_history(self, history_raw, history_mask, continuous_states=None, continuous_mask=None,
                       history_anchor=None, interest_anchor=None, interest_anchor_mask=None,
                       interest_codes=None, interest_mask=None, interest_channels=None):
        batch, length = history_mask.shape
        tokens = self.embed_sid(history_raw)
        if self.phi:
            if history_anchor is None:
                raise ValueError("Tiger phi requires per-item anchors")
            if self.hierarchical:
                pair=torch.stack([self.fine_to_coarse[history_anchor],history_anchor],dim=-1)
                anchors=self.anchor_embeddings_of(pair)
            else:
                anchors = self.anchor_embeddings_of(history_anchor).unsqueeze(2)
            codes = tokens
            separators = self.sep_token.view(1, 1, 1, -1).expand(batch, length, 1, -1)
            tokens = torch.cat((anchors, codes, separators), dim=2).flatten(1, 2)
            mask = history_mask.repeat_interleave(self.g_levels + codes.shape[2] + 1, dim=-1)
            if self.phi_pg:
                if not self.pg_user_prefix:
                    return self.encoder(inputs_embeds=tokens, attention_mask=mask).last_hidden_state, mask
                if interest_codes is None or interest_mask is None:
                    raise ValueError('PG requires user interest codes and mask')
                if self.hierarchical:
                    pair=torch.stack([self.fine_to_coarse[interest_codes],interest_codes],dim=-1)
                    prefix=(self.anchor_embeddings_of(pair)+self.interest_slot(
                        torch.arange(4,device=interest_codes.device))[None,:,None,:]).flatten(1,2)
                    interest_mask=interest_mask.repeat_interleave(2,dim=-1)
                else:
                    prefix = self.interest_embeddings(interest_codes) + self.interest_slot(
                        torch.arange(4, device=interest_codes.device))[None]
                tokens = torch.cat((prefix, tokens), dim=1)
                mask = torch.cat((interest_mask.bool(), mask), dim=1)
                return self.encoder(inputs_embeds=tokens, attention_mask=mask).last_hidden_state, mask
            return self._encode_prefix(tokens, mask, continuous_states, continuous_mask,
                                      interest_anchor, interest_anchor_mask)
        separators = self.sep_token.view(1, 1, 1, -1).expand(batch, length, 1, -1)
        tokens = torch.cat((tokens, separators), dim=2).flatten(1, 2)
        mask = history_mask.repeat_interleave(5, dim=-1)
        if self.phi_v8:
            if interest_codes is None or interest_mask is None:
                raise ValueError("v8 interest codes/mask required")
            prefix = self.interest_embeddings(interest_codes) + self.interest_slot(
                torch.arange(4, device=interest_codes.device))[None]
            tokens = torch.cat((prefix, tokens), dim=1)
            mask = torch.cat((interest_mask.bool(), mask), dim=1)
        if self.phi_v7:
            if interest_codes is None or interest_mask is None or interest_channels is None:
                raise ValueError('v7 interest codes/masks/channels required')
            prefix = (self.interest_embeddings(interest_codes) + self.interest_channel(interest_channels)
                      + self.interest_slot(torch.arange(4, device=interest_codes.device))[None])
            tokens = torch.cat((prefix, tokens), dim=1)
            mask = torch.cat((interest_mask.bool(), mask), dim=1)
        if self.phi_pred:
            if interest_codes is None or interest_mask is None:
                raise ValueError("the predictive-vocabulary mode requires per-slot interest codes")
            prefix = self.interest_embeddings(interest_codes.long())
            tokens = torch.cat((prefix, tokens), dim=1)
            mask = torch.cat((interest_mask.bool(), mask), dim=1)
        if self.deyi_enabled:
            if continuous_states is None or continuous_mask is None:
                raise ValueError("DeYi states are required by this model")
            slots = torch.arange(continuous_states.shape[1], device=continuous_states.device)
            state_tokens = self.continuous_state_projection(
                self.continuous_state_norm(continuous_states.to(self.continuous_state_projection.weight.dtype))
            )
            state_tokens = state_tokens + self.continuous_state_slots(slots).unsqueeze(0)
            tokens = torch.cat((state_tokens, tokens), dim=1)
            mask = torch.cat((continuous_mask.bool(), mask), dim=1)
        if self.dummy_memory:
            dummy = self.dummy_memory_token.view(1, self.dummy_slots, -1).expand(
                batch, self.dummy_slots, -1).to(tokens.dtype)
            tokens = torch.cat((dummy, tokens), dim=1)
            mask = torch.cat((mask.new_ones((batch, self.dummy_slots)), mask), dim=1)
        return self.encoder(inputs_embeds=tokens, attention_mask=mask).last_hidden_state, mask

    def _encode_prefix(self, tokens, mask, continuous_states, continuous_mask,
                       interest_anchor, interest_anchor_mask):
        """phi history was already assembled; prepend the per-slot anchors and DeYi states."""
        if interest_anchor is not None:
            slots = int(interest_anchor.shape[1])
            # The pair is (region, first codebook digit) -- the item code's first two tokens, so a
            # slot reads as a coarse item.
            regions = self.anchor_embeddings_of(interest_anchor[..., 0])
            # ``embed_sid`` takes [N, L] digits; the slot's digit is one position, so the extra
            # axis it returns is squeezed away before the pair is stacked.
            digits = self.embed_sid(interest_anchor[..., 1].unsqueeze(-1)).squeeze(-2)
            pair = torch.stack((regions, digits), dim=2)
            sep = self.sep_token.view(1, 1, 1, -1).expand(len(tokens), slots, 1, -1)
            prefix = torch.cat((pair, sep), dim=2).flatten(1, 2)      # [B, K*3, d]
            valid = (interest_anchor_mask.bool() if interest_anchor_mask is not None
                     else torch.ones_like(interest_anchor[..., 0], dtype=torch.bool))
            prefix_mask = valid.repeat_interleave(3, dim=-1)
            tokens = torch.cat((prefix, tokens), dim=1)
            mask = torch.cat((prefix_mask, mask), dim=1)
        if self.deyi_enabled:
            if continuous_states is None or continuous_mask is None:
                raise ValueError("DeYi states are required by this model")
            slots = torch.arange(continuous_states.shape[1], device=continuous_states.device)
            state_tokens = self.continuous_state_projection(
                self.continuous_state_norm(
                    continuous_states.to(self.continuous_state_projection.weight.dtype)))
            state_tokens = state_tokens + self.continuous_state_slots(slots).unsqueeze(0)
            tokens = torch.cat((state_tokens, tokens), dim=1)
            mask = torch.cat((continuous_mask.bool(), mask), dim=1)
        return self.encoder(inputs_embeds=tokens, attention_mask=mask).last_hidden_state, mask

    def decode_chain(self, hidden, mask, anchor_prev, code_prev):
        """phi decoder input for a (possibly partial) chain ``[g1, g2, c0..c3]``.

        Teacher forcing passes two anchors and three digits; beam search passes the prefix built
        so far, so either part may be empty.
        """
        bos = self.bos_token.unsqueeze(0).expand(len(hidden), 1, -1)
        pieces = [bos]
        if anchor_prev is not None and anchor_prev.shape[1]:
            if self.phi_pred:
                # The chain's first position is the predictive *route*, a shared token rather than
                # a property of the item.
                pieces.append(self.route_embeddings(anchor_prev.long()))
            else:
                pieces.append(self.anchor_embeddings_of(anchor_prev))
        if code_prev is not None and code_prev.shape[1]:
            pieces.append(self.embed_sid(code_prev))
        tokens = torch.cat(pieces, dim=1)
        return self.decoder(inputs_embeds=tokens, encoder_hidden_states=hidden,
                            encoder_attention_mask=mask, use_cache=False).last_hidden_state

    def decode(self, hidden, mask, prefix):
        bos = self.bos_token.unsqueeze(0).expand(len(prefix), 1, -1)
        tokens = torch.cat((bos, self.embed_sid(prefix)), dim=1)
        return self.decoder(inputs_embeds=tokens, encoder_hidden_states=hidden,
                            encoder_attention_mask=mask, use_cache=False).last_hidden_state

    def head_logits(self, decoded, depth):
        """Logits of hierarchy ``depth`` from a decode pass whose position ``depth`` is the last."""
        return self.chain_head(depth)(decoded[:, depth])

    def step_logits(self, decoded, depth):
        """Logits of hierarchy ``depth`` for beam search: last position of ``[BOS, c0..c_{d-1}]``."""
        return self.chain_head(depth)(decoded[:, -1])

    def forward(self, history_raw, history_mask, target_raw, target_owner,
                continuous_states=None, continuous_mask=None, history_anchor=None,
                interest_anchor=None, interest_anchor_mask=None, target_anchor=None,
                interest_codes=None, interest_mask=None, interest_channels=None,
                target_route=None, target_route_mask=None):
        hidden, mask = self.encode_history(
            history_raw, history_mask, continuous_states, continuous_mask,
            history_anchor, interest_anchor, interest_anchor_mask,
            interest_codes, interest_mask, interest_channels)
        if self.phi_pred and target_route is not None:
            return self.chain_forward(hidden, mask, target_owner, target_raw, target_route)
        # ``route_mode="none"`` reaches the native chain below: same memory in the encoder, no
        # route position to score.
        if self.phi:
            if target_anchor is None:
                raise ValueError("Tiger phi requires target anchors")
            return self.chain_forward(hidden, mask, target_owner, target_raw, target_anchor)
        # Each target is a separate four-digit sequence, not ten concatenated SIDs.
        owner_hidden = hidden.index_select(0, target_owner)
        owner_mask = mask.index_select(0, target_owner)
        # The route-free arm (§16.1 "memory only") keeps the phi encoder but scores the *native*
        # four-digit chain, so it must bypass ``chain_head(0)``: that is the route head with
        # ``vocabulary + 1`` classes, and mixing it with the 2048-way SID heads cannot be stacked.
        head_logits = ((lambda decoded, depth: self.output_heads[depth](decoded[:, depth]))
                       if self.phi_pred else self.head_logits)
        prefix = target_raw[:, :3]
        if self.training and self.prefix_sampling > 0:
            # Exposure-bias mitigation: training conditions on gold prefixes while beam search
            # conditions on its own, so a fraction of prefixes is replaced by the model's argmax.
            first = self.decode(owner_hidden, owner_mask, prefix)
            predicted = torch.stack([head_logits(first, depth).argmax(-1) for depth in range(3)],
                                    dim=1)
            keep_gold = torch.rand_like(prefix, dtype=torch.float32) >= self.prefix_sampling
            prefix = torch.where(keep_gold, prefix, predicted)
        decoded = self.decode(owner_hidden, owner_mask, prefix)
        return torch.stack([head_logits(decoded, depth) for depth in range(4)], dim=1)


    def chain_forward(self, hidden, mask, target_owner, target_raw, target_anchor):
        """phi teacher forcing: the region first, then the four codebook digits (v6: 5 positions)."""
        owner_hidden = hidden.index_select(0, target_owner)
        owner_mask = mask.index_select(0, target_owner)
        decoded = self.decode_chain(owner_hidden, owner_mask, target_anchor, target_raw[:, :3])
        return [self.head_logits(decoded, depth) for depth in range(4+self.g_levels)]


def row_losses(logits, targets, owners, rows):
    """Digit mean -> positive mean -> user mean; all positives participate."""
    positive = nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                                           reduction="none").view_as(targets).mean(1)
    sums = positive.new_zeros(rows).scatter_add_(0, owners, positive)
    return sums / torch.bincount(owners, minlength=rows)


def row_losses_chain(logits, targets, owners, rows):
    """Same reduction as ``row_losses`` for a chain whose depths have different vocabularies."""
    positive = torch.stack([
        nn.functional.cross_entropy(logits[depth].float(), targets[:, depth], reduction="none")
        for depth in range(len(logits))], dim=1).mean(1)
    sums = positive.new_zeros(rows).scatter_add_(0, owners, positive)
    return sums / torch.bincount(owners, minlength=rows)
