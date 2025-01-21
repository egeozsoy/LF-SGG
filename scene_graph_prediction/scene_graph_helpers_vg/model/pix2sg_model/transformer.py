# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.detr_configs import N_PREDS, N_ENTITIES, ENTITY_START, PRED_START, INSTANCE_START, \
    ASSUMED_MAX_INSTANCE_ID, TEMPERATURE, TOP_K, TOP_P


class DecoderEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_dim
        )

        self.LayerNorm = torch.nn.LayerNorm(
            hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.embedding = DecoderEmbeddings(800, d_model, 799, 1600, dropout)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        #        self.token_drop = SpatialDropout(drop=0.5)
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def top_k_top_p_filtering(
            self,
            logits: torch.FloatTensor,
            top_k: int = 0,
            top_p: float = 1.0,
            filter_value: float = -float("Inf"),
            min_tokens_to_keep: int = 1,
    ) -> torch.FloatTensor:
        """
        https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317#file-top-k-top-p-py-L42
        https://github.com/umiswing/top_k_top_p_filtering/blob/main/top-k-top-p.py
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            top_k (`int`, *optional*, defaults to 0):
                If > 0, only keep the top k tokens with highest probability (top-k filtering)
            top_p (`float`, *optional*, defaults to 1.0):
                If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
                filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            min_tokens_to_keep (`int`, *optional*, defaults to 1):
                Minimumber of tokens we keep per batch example in the output.
        """
        top_p = float(top_p)
        if top_k > 0:
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, filter_value)

        if 0 < top_p <= 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, filter_value)

        return logits

    def forward(self, image_src, image_mask, image_pos_embed, seq, vocab_embed, max_num_rels=None):
        # flatten NxCxHxW to HWxNxC
        REAL_BATCH_SIZE = seq.shape[0]
        _, _, h, w = image_src.shape
        image_src = image_src.flatten(2).permute(2, 0, 1)
        image_pos_embed = image_pos_embed.flatten(2).permute(2, 0, 1)
        image_mask = image_mask.flatten(1)

        tgt = self.embedding(seq).permute(1, 0, 2)
        #        tgt = self.token_drop(self.embedding(seq), noise_shape=(bs, 501, 1)).permute(1, 0, 2)
        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
        query_embed = query_embed.repeat(1, REAL_BATCH_SIZE, 1)
        src = image_src
        mask = image_mask
        pos_embed = image_pos_embed
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed.half())

        if self.training:
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed[:len(tgt)],
                              tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device))
            return hs.transpose(1,
                                2), None  # memory.permute(1, 2, 0).reshape(bs, c, h, w)
        else:
            MAX_NUM_RELS = 100 if max_num_rels is None else max_num_rels
            # Rel Format: [sub_id,sub_instance,obj_id,obj_instance,pred_id]
            TOKENS_PER_REL = 5
            ITERATON_STEPS = MAX_NUM_RELS * TOKENS_PER_REL
            index_to_type = {}
            for i in range(ITERATON_STEPS):
                if i % TOKENS_PER_REL in [0, 2]:
                    index_to_type[i] = 'entity_id'
                elif i % TOKENS_PER_REL in [4]:
                    index_to_type[i] = 'rel_id'
                elif i % TOKENS_PER_REL in [1, 3]:
                    index_to_type[i] = 'entity_instance'
                else:
                    raise ValueError('Invalid index')

            values = []
            # token_freq = torch.zeros((REAL_BATCH_SIZE, vocab_embed.layers[-1].out_features), device=seq.device)
            all_attn_maps = []
            for i in range(ITERATON_STEPS):
                tgt = self.embedding(seq).permute(1, 0, 2)
                query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
                query_embed = query_embed.repeat(1, REAL_BATCH_SIZE, 1)
                hs, attn_maps = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                             pos=pos_embed, query_pos=query_embed[:len(tgt)],
                                             tgt_mask=generate_square_subsequent_mask(len(tgt)).to(tgt.device), return_attention_maps=True)
                all_attn_maps.append(attn_maps[0][:, -1, :])
                out = vocab_embed(hs.transpose(1, 2)[-1, :, -1, :])

                if index_to_type[i] == 'rel_id':  # in vocab, rel_ids are between 0-50
                    out = out.softmax(-1)
                    out = out[:, PRED_START:PRED_START + N_PREDS]
                    value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
                elif index_to_type[i] == 'entity_id':  # in vocab, entity_ids are between 51-201
                    # adjust scores to avoid repeating the same entity
                    out = out[:, ENTITY_START:ENTITY_START + N_ENTITIES]
                    out = out / TEMPERATURE
                    out = self.top_k_top_p_filtering(out, top_k=TOP_K, top_p=TOP_P)
                    out = out.softmax(-1)
                    extra_seq = torch.multinomial(out, 1)
                    value = out.gather(-1, extra_seq)
                else:
                    out = out.softmax(-1)
                    out = out[:, INSTANCE_START:INSTANCE_START + ASSUMED_MAX_INSTANCE_ID]
                    value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
                if index_to_type[i] == 'entity_id':
                    extra_seq += ENTITY_START
                    # token_freq[:, extra_seq.squeeze(-1)] += 1
                elif index_to_type[i] == 'entity_instance':
                    extra_seq += INSTANCE_START

                seq = torch.cat([seq, extra_seq], dim=-1)
                values.append(value)
            return seq, torch.cat(values, dim=-1), all_attn_maps


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, return_attention_maps=False):
        output = tgt
        attn_maps = []

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, return_attention_maps=return_attention_maps)
            if return_attention_maps:
                attn_maps.append(output[1])
                output = output[0]
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if return_attention_maps:
                return torch.stack(intermediate), attn_maps
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None, return_attention_maps=False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        if return_attention_maps:
            tgt2, attn = tgt2
        else:
            tgt2 = tgt2[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attention_maps:
            return tgt, attn
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, return_attention_maps=False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attention_maps=return_attention_maps)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attention_maps=return_attention_maps)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
