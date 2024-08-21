import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn, einsum
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
import torch

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_bloom import BloomConfig

logger = logging.get_logger(__name__)

from transformers.models.bloom.modeling_bloom import (
    BloomPreTrainedModel,
    BloomBlock, BloomModel, BloomForCausalLM,
    BloomMLP, BloomAttention, _make_causal_mask, _expand_mask, build_alibi_tensor,
)
from transformers import AutoTokenizer, AutoModel

import pickle


class VectorInfusedBloomModelConfig(BloomConfig):
    def __init__(self, vector_dim=768, use_vector_num=1, stride=1, online=0, train_encoder=True, train_decoder=True, **kwargs):
        super(VectorInfusedBloomModelConfig, self).__init__(**kwargs)
        self.use_vector_num = use_vector_num
        self.vector_dim = vector_dim
        self.stride = stride
        self.online = online # deprecated
        self.train_encoder = train_encoder
        self.train_decoder = train_decoder


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class MaskedCrossAttention(nn.Module):
    """
    Input shape:
        x: (batch_size B, seq_len L, hidden_size H)
        media (perceived): (B, num_media M, num_latent_vector T, H)
    Output shape:
        (B, L, dim_head * num_heads)
    """

    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, media):
        b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, 'b t n d -> b (t n) d')

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    """
    x, media -> [MaskedCrossAttention] -> [FFW] -> output

    Input shape:
        x: (batch_size B, seq_len L, hidden_size H)
        media (perceived): (B, num_media M, num_latent_vector T, H)
    Output shape:

    """

    def __init__(self, *, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, media, ):
        # input:
        x = self.attn(x, media) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


class VectorInfusedBloomBlock(nn.Module):
    def __init__(self, config: VectorInfusedBloomModelConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # Vector Projection Layer
        self.vector_dim = config.vector_dim
        self.vector_projector = nn.Linear(self.vector_dim, hidden_size, bias=True)

        if '1b' in config.name_or_path:
            ff_mult = 4
        elif '3b' in config.name_or_path:
            ff_mult = 2
        else:
            ff_mult = 4
        
        print("ff_mult: ", ff_mult)

        self.vector_infused_attn = GatedCrossAttentionBlock(dim=hidden_size, dim_head=hidden_size // 8, heads=8,
                                                            ff_mult=ff_mult)
        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head

        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

        # To resolve conflicts of new and old codebases.
        try:
            train_encoder = self.config.train_encoder
        except AttributeError:
            train_encoder = False

        if train_encoder:
            print("Init cross-attn with params in self-attn")
            self.vector_infused_attn.attn.to_q.weight.data = self.self_attention.query_key_value.weight[:hidden_size, :].data
            # self.vector_infused_attn.to_q.bias.data = self.self_attention.query_key_value.bias[:hidden_size].data

            self.vector_infused_attn.attn.to_kv.weight.data = self.self_attention.query_key_value.weight[hidden_size:, :].data
            # self.vector_infused_attn.to_kv.bias.data = self.self_attention.query_key_value.bias[hidden_size:].data

            self.vector_infused_attn.attn.to_out.weight.data = self.self_attention.dense.weight.data
            # self.vector_infused_attn.to_out.bias.data = self.self_attention.dense.bias.data


    def forward(self,
                hidden_states: torch.Tensor,
                alibi: torch.Tensor,
                attention_mask: torch.Tensor,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                head_mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                output_attentions: bool = False,
                infused_vectors: Optional[torch.Tensor] = None,
                ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # vector infused input
        if self.config.use_vector_num > 0 and infused_vectors is not None:
            infused_vectors = infused_vectors[:, :self.config.use_vector_num:self.config.stride, :]
            infused_vectors = self.vector_projector(infused_vectors)
            infused_vectors = infused_vectors.unsqueeze(-2)
            infused_hidden_states = self.vector_infused_attn(hidden_states, infused_vectors)
        else:
            infused_hidden_states = hidden_states

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(infused_hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = infused_hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class VectorInfusedBloomModel(BloomPreTrainedModel):
    def __init__(self, config: VectorInfusedBloomModelConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([VectorInfusedBloomBlock(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
            self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            infused_vectors: Optional[torch.Tensor] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(inputs_embeds.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    infused_vectors=infused_vectors,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class VectorInfusedBloomForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", 
                                       r"lm_head.weight",
                                       r'h.*.vector_infused_attn.*',
                                       r'h.*.vector_projector.*',
                                       r'encoder.*',
                                       ]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BloomConfig):
        super().__init__(config)
        # self.encoder = AutoModel.from_pretrained("bert-base-uncased")

        self.encoder = pickle.load(open('/mnt/nas-alinlp/zhuochen.zc/others/triviaqa-unfiltered/bert-base-uncased-model.pkl', 'rb'))

        self.transformer = VectorInfusedBloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # if hasattr(self.config, 'train_encoder'):
        #     print('Setting freezing/training parts..')
        #     print('Encoder training: ', self.config.train_encoder)
        #     print('Decoder training: ', self.config.train_decoder)
        #     self.encoder.requires_grad_(self.config.train_encoder)
        #     self.transformer.requires_grad_(self.config.train_decoder)
        
        # self.lm_head.requires_grad_(True)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            # infused_vectors: Optional[torch.Tensor] = None, deprecated
            ctxs_online_input_ids: Optional[torch.Tensor] = None,
            ctxs_online_token_type_ids: Optional[torch.Tensor] = None,
            ctxs_online_attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # "infused_vectors": infused_vectors,
                "ctxs_online_input_ids": ctxs_online_input_ids,
                "ctxs_online_token_type_ids": ctxs_online_token_type_ids,
                "ctxs_online_attention_mask": ctxs_online_attention_mask,
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # infused_vectors: Optional[torch.Tensor] = None, deprecated
            ctxs_online_input_ids: Optional[torch.Tensor] = None,
            ctxs_online_token_type_ids: Optional[torch.Tensor] = None,
            ctxs_online_attention_mask: Optional[torch.Tensor] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Run encoder forward() to get online infused_vectors with gradient
        # if hasattr(self.config, 'use_vector_num'): # Error from old configs
        if self.config.use_vector_num > 0:
            if ctxs_online_input_ids is not None:
                bz, num_vec_each, seq_len = ctxs_online_input_ids.shape
                ctxs_online_input_ids = ctxs_online_input_ids.reshape(-1, seq_len)
                ctxs_online_token_type_ids = ctxs_online_token_type_ids.reshape(-1, seq_len)
                ctxs_online_attention_mask = ctxs_online_attention_mask.reshape(-1, seq_len)

                ctxs_online_input_ids = ctxs_online_input_ids.to(self.encoder.device)
                ctxs_online_token_type_ids = ctxs_online_token_type_ids.to(self.encoder.device)
                ctxs_online_attention_mask = ctxs_online_attention_mask.to(self.encoder.device)
                
                # breakpoint()

                infused_vectors = self.encoder(ctxs_online_input_ids, ctxs_online_token_type_ids, ctxs_online_attention_mask)
                infused_vectors = infused_vectors.last_hidden_state[:, 0, :]
                infused_vectors = infused_vectors.reshape(bz, num_vec_each, -1)
            else:
                print("config.online > 0 but no ctxs_online_input_ids!")
                exit(-1)
            
        transformer_outputs = self.transformer(
            input_ids,
            infused_vectors=infused_vectors,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
            self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)
