from config.train_options import TrainOptions

from config.model.deepseek_config import DeepSeekV2


class DeepSeekProjection:
    def __init__(self, model_config: DeepSeekV2, train_options: TrainOptions):
        self._model_config = model_config
        self._train_options = train_options

        # for internal usage
        (
            self._vocab_size,
            self._max_seq_len,
            self._dim,
            self._intermediate_size,
            self._moe_intermediate_size,
            self._n_layers,
            self._n_dense_layers,
            self._n_heads,
            self._qk_nope_head_dim,
            self._q_lora_rank,
            self._kv_lora_rank,
            self._qk_rope_head_dim,
            self._n_experts_shared,
            self._n_experts_routed,
            self._router_top_k,
        ) = (
            model_config.vocab_size,
            model_config.max_seq_len,
            model_config.dim,
            model_config.intermediate_size,
            model_config.moe_intermediate_size,
            model_config.n_layers,
            model_config.n_dense_layers,
            model_config.n_heads,
            model_config.qk_nope_head_dim,
            model_config.q_lora_rank,
            model_config.kv_lora_rank,
            model_config.qk_rope_head_dim,
            model_config.n_experts_shared,
            model_config.n_experts_routed,
            model_config.router_top_k,
        )

        self._causal_mask = train_options.causal_mask
        self._fused_atten = train_options.fused_atten
        self._bytes_of_dtype = train_options.use_dtype.bytes_of_dtype()

    def get_num_params(self, include_embedding: bool = True) -> tuple[int, int]:
        num_params_embedding = 0
        if include_embedding:
            num_params_embedding = (
                self._vocab_size
                * self._dim  # Word Token Embedding(WTE)
                # + self._max_seq_len * self._dim  # Word Position Embedding (WPE)
            )

        # MLA projection for Q, K and V
        if self._q_lora_rank is None:
            num_params_proj_q = (
                self._dim
                * self._n_heads
                * (self._qk_nope_head_dim + self._qk_rope_head_dim)
            )
        else:
            num_params_down_q = self._dim * self._q_lora_rank
            num_params_up_q = self._q_lora_rank * self._n_heads * self._qk_nope_head_dim
            num_params_rope_q = (
                self._q_lora_rank * self._n_heads * self._qk_rope_head_dim
            )
            num_params_proj_q = num_params_down_q + num_params_up_q + num_params_rope_q
        num_params_down_kv = self._dim * self._kv_lora_rank
        num_params_up_k = self._kv_lora_rank * self._n_heads * self._qk_nope_head_dim
        num_params_rope_k = self._dim * self._qk_rope_head_dim
        num_params_up_v = self._kv_lora_rank * self._n_heads * self._qk_nope_head_dim
        # out proj
        num_params_o = (
            self._n_heads
            * self._qk_nope_head_dim  # v_head_dim = qk_nope_head_dim
            * self._dim
        )
        num_params_atten = (
            num_params_proj_q
            + num_params_down_kv
            + num_params_up_k
            + num_params_rope_k
            + num_params_up_v
            + num_params_o
        )

        num_params_ffn = self._dim * self._moe_intermediate_size * 3
        num_params_ffn_dense = self._dim * self._intermediate_size * 3
        # MoE, the sparse param count
        num_params_gate = 0
        n_experts = self._n_experts_routed + self._n_experts_shared
        num_params_ffn_activated = num_params_ffn
        if n_experts > 1:
            num_params_gate = self._dim * self._n_experts_routed
            num_params_ffn *= n_experts
            num_params_ffn_activated *= self._n_experts_shared + self._router_top_k

        num_params_norm = 2 * self._dim
        # additional RMSNorm after the compressed latent vectors
        num_params_norm += (
            self._kv_lora_rank + 0 if self._q_lora_rank is None else self._q_lora_rank
        )

        num_params_final_norm = self._dim

        num_params = (
            num_params_embedding
            + self._n_dense_layers
            * (num_params_atten + num_params_norm + num_params_ffn_dense)
            + (self._n_layers - self._n_dense_layers)
            * (num_params_atten + num_params_norm + num_params_ffn + num_params_gate)
            + num_params_final_norm
        )

        num_params_activated = (
            num_params_embedding
            + self._n_dense_layers
            * (num_params_atten + num_params_norm + num_params_ffn_dense)
            + (self._n_layers - self._n_dense_layers)
            * (
                num_params_atten
                + num_params_norm
                + num_params_ffn_activated
                + num_params_gate
            )
            + num_params_final_norm
        )
        return num_params, num_params_activated

    def get_num_flop_fwd(self, batch_size: int) -> int:
        # MLA projection of Q, K and V
        if self._q_lora_rank is None:
            num_flop_proj_q = (
                2
                * batch_size
                * self._max_seq_len
                * self._dim
                * self._n_heads
                * (self._qk_nope_head_dim + self._qk_rope_head_dim)
            )
        else:
            num_flop_down_q = (
                2 * batch_size * self._max_seq_len * self._dim * self._q_lora_rank
            )
            num_flop_up_q = (
                2
                * batch_size
                * self._max_seq_len
                * self._q_lora_rank
                * self._qk_nope_head_dim
                * self._n_heads
            )
            num_flop_rope_q = (
                2
                * batch_size
                * self._max_seq_len
                * self._q_lora_rank
                * self._qk_rope_head_dim
                * self._n_heads
            )
            num_flop_proj_q = num_flop_down_q + num_flop_up_q + num_flop_rope_q
        num_flop_down_k = (
            2 * batch_size * self._max_seq_len * self._dim * self._kv_lora_rank
        )
        num_flop_up_k = (
            2
            * batch_size
            * self._max_seq_len
            * self._kv_lora_rank
            * self._qk_nope_head_dim
            * self._n_heads
        )
        num_flop_rope_k = (
            2 * batch_size * self._max_seq_len * self._dim * self._qk_rope_head_dim
        )
        num_flop_proj_k = num_flop_down_k + num_flop_up_k + num_flop_rope_k
        num_flop_proj_v = (
            2
            * batch_size
            * self._max_seq_len
            * self._qk_nope_head_dim
            * self._n_heads
            * self._dim
        )
        num_flop_qkv_proj = num_flop_proj_q + num_flop_proj_k + num_flop_proj_v

        # see the discussion: https://github.com/pytorch/torchtitan/pull/280
        num_flop_sdpa = 4 * batch_size * self._max_seq_len**2 * self._dim
        num_flop_sdpa //= 2 if self._causal_mask else 1
        num_flop_out_proj = 2 * batch_size * self._max_seq_len * self._dim**2
        num_flop_fwd_atten = num_flop_qkv_proj + num_flop_sdpa + num_flop_out_proj

        num_flop_fwd_ffn = (
            2 * batch_size * self._max_seq_len * self._dim * self._moe_intermediate_size
        ) * 3
        num_flop_fwd_ffn_dense = (
            2 * batch_size * self._max_seq_len * self._dim * self._intermediate_size
        ) * 3
        # MoE, the active param
        n_experts = self._n_experts_shared + self._n_experts_routed
        if n_experts > 1:
            num_flop_fwd_ffn *= (
                self._n_experts_shared + self._router_top_k
            )  # num of activated experts
            num_flop_gate = (
                2 * batch_size * self._max_seq_len * self._dim * self._n_experts_routed
            )
            num_flop_fwd_ffn += num_flop_gate

        num_flop_fwd_logits = (
            2 * batch_size * self._max_seq_len * self._dim * self._vocab_size
        )

        return (
            self._n_dense_layers * (num_flop_fwd_atten + num_flop_fwd_ffn_dense)
            + (self._n_layers - self._n_dense_layers)
            * (num_flop_fwd_atten + num_flop_fwd_ffn)
            + num_flop_fwd_logits
        )

    def get_num_flop_per_token(self):
        batch_size = 1  # dummy
        num_flop_per_token = (
            self.get_num_flop_fwd(batch_size)
            / batch_size
            / self._max_seq_len
            * 3  # bwd = 2 * fwd
        )
        return num_flop_per_token

    def _get_num_flop_QK_fwd(self, batch_size: int) -> int:
        """
        Forward FLOPs for QK^T of all chunked transformer blocks, which is re-computed on backward by Flash attention
        """
        num_flop_qk = self._n_layers * (
            2 * batch_size * self._max_seq_len**2 * self._dim
        )
        num_flop_qk //= 2 if self._causal_mask else 1
        return num_flop_qk

    def get_num_flop_bwd(self, batch_size: int) -> int:
        num_flop_fwd = self.get_num_flop_fwd(batch_size)
        num_flop_bwd = num_flop_fwd * 2
        # Flash-attention uses re-computation for QK^T
        if self._fused_atten:
            qk_fwd_flop = self._get_num_flop_QK_fwd(batch_size)
            num_flop_bwd += qk_fwd_flop

        return num_flop_bwd
