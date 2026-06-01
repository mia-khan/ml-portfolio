import torch
import heapq
import copy
import math

import os
from torch import nn
from d2l import torch as d2l
from tsv_seq2seq_data import TSVSeq2SeqData
from torch.cuda.amp import autocast

#############################################################################################################################
########## PB1A : cross attention for RNN Decoder (queries) -> RNN Encoder (key, values) ###########################################

class MaskedSoftmax(nn.Module):
    def forward(self, X, valid_lens):
        if valid_lens is None:
            return torch.softmax(X, dim=-1)
        shape = X.shape
        # Repeat or reshape the valid lengths so every query/key pair has a mask entry.
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = X.reshape(-1, shape[-1])
        max_len = X.size(1)
        # Build a boolean mask where False entries represent padded positions.
        mask = torch.arange(max_len, device=X.device)[None, :] < valid_lens[:, None]
        X = X.masked_fill(~mask, -1e6)
        # Driving padded logits toward -inf guarantees they become exact zeros after softmax.
        return torch.softmax(X.reshape(shape), dim=-1)



# class AdditiveAttention(nn.Module):
#     """Bahdanau-style additive attention built on the local MaskedSoftmax."""
#     def __init__(self, key_size, query_size, num_hiddens, dropout=0.0):
#         super().__init__()
#         self.W_k = nn.LazyLinear(num_hiddens, bias=False)
#         self.W_q = nn.LazyLinear(num_hiddens, bias=False)
#         self.w_v = nn.LazyLinear(1, bias=False)
#         self.dropout = nn.Dropout(dropout)
#         self._masked_softmax = MaskedSoftmax()

#     def forward(self, queries, keys, values, valid_lens=None):
#         # Transform queries/keys to common dimensionality
#         queries = self.W_q(queries)                        # (batch, num_queries, num_hiddens)
#         keys = self.W_k(keys)                              # (batch, num_kv, num_hiddens)
#         # Broadcast addition so every query pairs with every key
#         features = torch.tanh(
#             queries.unsqueeze(2) + keys.unsqueeze(1)       # (batch, num_queries, num_kv, num_hiddens)
#         )
#         scores = self.w_v(features).squeeze(-1)            # (batch, num_queries, num_kv)

#         weights = self._masked_softmax(scores, valid_lens) # (batch, num_queries, num_kv)
#         self.attention_weights = weights
#         weights = self.dropout(weights)
#         return torch.bmm(weights, values)                  # (batch, num_queries, value_dim)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.masked_softmax = MaskedSoftmax()

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Scale the dot product by sqrt(d) to keep gradients stable across head sizes.
        # Compute attention scores by multiplying queries with transposed keys and scaling by sqrt(d)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # Masked softmax prevents padded tokens from contributing probability mass.
        # Derive attention_weights by applying the masked softmax to the scores while honoring valid lengths
        attention_weights = self.masked_softmax(scores, valid_lens)
        self.attention_weights = attention_weights
        # Weighted sum of values uses attention weights as coefficients across the sequence dimension.
        # Compute output by taking the weighted combination of the values according to attention_weights
        output = self.dropout(torch.bmm(attention_weights, values))
        return output 
    
class SingleHeadAttention(nn.Module):
    """DotProd-style attention built on the local MaskedSoftmax."""
    def __init__(self, key_size, query_size, num_hiddens, dropout=0.0):
        super().__init__()
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.W_v = nn.LazyLinear(num_hiddens, bias=False)
        self.attention = DotProductAttention(dropout)
    def forward(self, queries, keys, values, valid_lens=None):
        # Transform queries/keys to common dimensionality
        queries = self.W_q(queries)                        # (batch, num_queries, num_hiddens)
        keys = self.W_k(keys)                              # (batch, num_kv, num_hiddens)
        values = self.W_v(values)        
        # Self.attention performs the actual similarity -> distribution -> weighted sum.
        output = self.attention(queries, keys, values, valid_lens)
        self.attention_weights = self.attention.attention_weights
        return output


class AttentionDecoder(d2l.Decoder):
    'Base class for attention decoders.'
    def __init__(self):
            super().__init__()

    @property
    def attention_weights(self):
            raise NotImplementedError

       

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        #self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)  # Bahdanau-style scorer
        self.attention = SingleHeadAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Convert token ids to vectors
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)  # Decoder consumes embedding+context at every step
        self.dense = nn.LazyLinear(vocab_size)  # Project hidden states to vocab logits
        self.apply(d2l.init_seq2seq)  # Initialize weights for stable training

    def init_state(self, enc_outputs, enc_valid_lens):
        # Shape of outputs: (num_steps, batch_size, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        # Transpose encoder outputs so batch dimension comes first (needed by attention)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # Shape of enc_outputs: (batch_size, num_steps, num_hiddens).
        # Shape of hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # Shape of the output X: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []  # Collect logits and attention maps per step
        for x in X:
            # Shape of query: (batch_size, 1, num_hiddens)
            # Decoder makes a prediction conditioned on its most recent hidden state.
            query = torch.unsqueeze(hidden_state[-1], dim=1)  # Use last GRU layer as query
            # Shape of context: (batch_size, 1, num_hiddens)
            query_fp32 = query.float()
            enc_outputs_fp32 = enc_outputs.float()

            # Run attention outside AMP so masked_softmax can write large negatives safely
            # Attention returns a context vector: convex combo of encoder states weighted by similarity to the query.
            # Produce a context vector by running attention over encoder outputs using the query,
            # then convert the high-precision result back to the original dtype
            context = self.attention(query_fp32, enc_outputs_fp32, enc_outputs_fp32, enc_valid_lens)
            context = context.to(query.dtype)
            
            # Concatenate on the feature dimension
            # Feeding both context and current embedding lets the GRU disambiguate lexical vs. contextual info.
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)  # Feed context + current embedding
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)  # Advance decoder GRU
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))  # Map GRU outputs to token logits
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights  # Expose collected attention weights



#############################################################################################################
#################  PB1B : MultiHead cross attention for RNN Decoder (queries) -> RNN Encoder (key, values)###########


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        # All heads share the same attention module; projections create per-head subspaces.
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens)
        self.W_k = nn.LazyLinear(num_hiddens)
        self.W_v = nn.LazyLinear(num_hiddens)
        self.W_o = nn.LazyLinear(num_hiddens)

    def forward(self, queries, keys, values, valid_lens=None):

        # Project to head-specific subspaces and split the batch across heads.
        # Apply W_q/W_k/W_v to queries/keys/values respectively,
        # then pass each projected tensor through _transpose_qkv to create (batch*num_heads, steps, head_dim)
        queries = self._transpose_qkv(self.W_q(queries))
        keys = self._transpose_qkv(self.W_k(keys))
        values = self._transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # Each head now produces its own context tensor; stacking heads increases modeling capacity.
        output = self.attention(queries, keys, values, valid_lens)
        output = self._transpose_output(output)
        return self.W_o(output)

    def _transpose_qkv(self, X):
        batch_size, num_steps, num_hiddens = X.shape
        X = X.reshape(batch_size, num_steps, self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        # Collapse batch/head so attention can process every head in parallel.
        return X.reshape(-1, num_steps, X.shape[-1])

    def _transpose_output(self, X):
        batch_size = X.shape[0] // self.num_heads
        num_steps = X.shape[1]
        X = X.reshape(batch_size, self.num_heads, num_steps, -1)
        X = X.permute(0, 2, 1, 3)
        # Merge head dimension back into the feature dimension.
        return X.reshape(batch_size, num_steps, -1)



class MultiHeadSeq2SeqDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 num_heads=4, dropout=0.0):
        super().__init__()
        
        # Instantiate MultiHeadAttention so the decoder can build richer context vectors
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout)
          
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            query = hidden_state[-1].unsqueeze(1)

            # Each head compares this query against encoder features, then the outputs are concatenated into one context vector.
            # Produce a multi-head context vector by matching the query against the encoder outputs
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)

            x = torch.cat((context, x.unsqueeze(1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(
                self.attention.attention.attention_weights.detach().cpu())
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

#######################################################################################################################
#################################### PB2A :  enc-self-attn  + cross_attn (Dec - Enc) ####################
class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class SelfAttentionAugmentedEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 num_heads=4, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout, batch_first=True)
        
        # Instantiate MultiHeadAttention so encoder outputs contextualize each other
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout)
        
        self.addnorm = AddNorm(num_hiddens, dropout)
        self.apply(d2l.init_seq2seq)

    def forward(self, X, valid_lens, *args):
        embs = self.embedding(X.type(torch.int64))
        outputs, state = self.rnn(embs)
        if valid_lens is not None:
            valid_lens = valid_lens.to(outputs.device)

        # Add self-attention with queries = val = keys = outputs
        # Each timestep attends over the entire encoder sequence to build a contextualized representation
        attn = self.attention(outputs, outputs, outputs, valid_lens)
        
        outputs = self.addnorm(outputs, attn)
        return outputs.permute(1, 0, 2), state



#################################################################################################################################
#####################################  PB1B :  dec-self-att  +   enc-self-attn   + cross_attn (Dec - Enc) ####################


class SelfAttentiveGRUDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 num_heads=4, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.self_attn = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.cross_attn = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.rnn = nn.GRU(num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X)
        batch_size, num_steps, _ = X.shape
        dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        
        # Each position in the partially generated target can only attend to its past tokens.
        # Perform masked self-attention over the decoder inputs using dec_valid_lens to enforce causality
        Z = self.self_attn(X, X, X, dec_valid_lens)

        Y = self.addnorm1(X, Z)
        
        # Cross-attention: let each decoder timestep gather information from the encoder sequence
        # while respecting enc_valid_lens
        context = self.cross_attn(Y, enc_outputs, enc_outputs, enc_valid_lens)

        Y = self.addnorm2(Y, context)
        Y = Y.permute(1, 0, 2)
        outputs = []
        for y in Y:
            out, hidden_state = self.rnn(y.unsqueeze(0), hidden_state)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self.cross_attn.attention.attention_weights

#################################################################################################################################
################################## PB3A  transformer-Decoder ; RNN-Encoder ##########################################
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_hiddens, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / num_hiddens))
        pe = torch.zeros(1, max_len, num_hiddens)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:, :X.size(1)].to(X.device)
        return self.dropout(X)
    
# class BatchFirstGRUEncoder(d2l.Encoder):
#     def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
#                  dropout=0.0):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
#                           dropout=dropout, batch_first=True)
#         self.apply(d2l.init_seq2seq)

#     def forward(self, X, *args):
#         embs = self.embedding(X.type(torch.int64))
#         outputs, state = self.rnn(embs)
#         return outputs
        
class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        batch_size, num_steps, _ = X.shape
        if self.training:
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device)
            dec_valid_lens = dec_valid_lens.repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        
        # First attention block is causal self-attention: each position builds context from already-decoded tokens only.
        # Run causal self-attention over the decoder states and apply the residual LayerNorm wrapper
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        
        # Second block is cross-attention: queries are the decoder states, keys/values are encoder outputs.
        # Perform cross-attention against encoder outputs followed by another residual LayerNorm
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        
        # Instantiate positional encoding module so token order is preserved
        # Without recurrence, transformers need explicit position information via sinusoidal encodings
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"block{i}", TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        if isinstance(enc_outputs, tuple):
            enc_outputs = enc_outputs[0]
        if enc_outputs.dim() == 3 and enc_valid_lens is not None:
            batch_dim = enc_valid_lens.shape[0]
            if enc_outputs.shape[0] != batch_dim and enc_outputs.shape[1] == batch_dim:
                # Some encoders output (time, batch, dim); Transformer blocks expect batch-first.
                enc_outputs = enc_outputs.permute(1, 0, 2)
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # Track attention weights separately for decoder self-attention (layer 0) and cross-attention (layer 1).
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    

    @property
    def attention_weights(self):
        return self._attention_weights



########################################################################################################
################################################ PB3B :  transformer Encoder ###########################

class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout)  # Self-attention over the sequence
        self.addnorm1 = AddNorm(num_hiddens, dropout)  # Residual + norm after attention
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)  # Token-wise MLP
        self.addnorm2 = AddNorm(num_hiddens, dropout)  # Residual + norm after FFN

    def forward(self, X, valid_lens):
        # First sublayer: multi-head self-attention with padding masks
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # Second sublayer: position-wise FFN followed by another residual path
        return self.addnorm2(Y, self.ffn(Y))



class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens  # Keep hidden size for later scaling
        self.embedding = nn.Embedding(vocab_size, num_hiddens)  # Token lookup table
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)  # Deterministic position features
        self.blks = nn.Sequential()
        for i in range(num_blks):
            # Stack identical encoder blocks to deepen the model
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Scale embeddings before adding bounded positional encodings
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)  # Cache attention maps for visualization
        for i, blk in enumerate(self.blks):
            # Each block is identical: self-attention followed by a position-wise feed-forward net.
            X = blk(X, valid_lens)  # Apply each encoder block with padding masks
            self.attention_weights[
                i] = blk.attention.attention.attention_weights  # Store the weights per block
        return X  # Return contextualized token representations







########################################################################################################
################################################  beam search for Seq2Seq models output ##########################################

def _clone_state(state):
    enc_outputs, enc_valid_lens, cache = state
    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
        else:
            # Clone ensures future beam expansions cannot overwrite past key/value tensors.
            new_cache.append(layer_cache.detach().clone())
    return [enc_outputs, enc_valid_lens, new_cache]


def beam_search_translate(model, src_tokens, data_module,
                          beam_size=3, max_steps=50, alpha=10, device=None):
    """
    alpha: length-penalty exponent (0 => no penalty, 0.6–1.0 typical)
    """
    if device is None:
        device = next(model.parameters()).device

    encoder_input = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)
    src_len = torch.tensor([len(src_tokens)], dtype=torch.long, device=device)

    enc_outputs = model.encoder(encoder_input, src_len)
    dec_state = model.decoder.init_state(enc_outputs, src_len)

    bos_id = data_module.tgt_vocab['<bos>']
    eos_id = data_module.tgt_vocab['<eos>']

    beams = [([bos_id], 0.0, dec_state)]
    completed = []

    for _ in range(max_steps):
        new_beams = []
        for tokens, score, state in beams:
            if tokens[-1] == eos_id:
                # Finished hypotheses are moved to completed list and never expanded again.
                completed.append((tokens, score))
                continue
            dec_input = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            logits, new_state = model.decoder(dec_input, _clone_state(state))
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
            for log_p, token_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    # Clone the decoder state so each child beam keeps its own cache/history.
                    child_state = _clone_state(new_state)
                    new_beams.append((tokens + [token_id], score + log_p, child_state))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if not beams:
            break

    completed.extend((tokens, score) for tokens, score, _ in beams)


    def length_penalty(tokens, score):
        L = max(len(tokens), 1)
        # Apply Google NMT length penalty so shorter beams are not unfairly favored.
        return score / (((5 + L) / 6) ** alpha)


    best_tokens, _ = max(completed, key=lambda x: length_penalty(*x))
    return best_tokens[1:]  # drop <bos>
