# MLA Projection



Ref

- [DeepSeek-V2/modeling_deepseek.py](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py)
- [MLA模块的计算过程](https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md#mla模块的计算过程)



## MLA

<img src="img/mla.png" width="600">

$$
\begin{align}
\mathbf{c}_{t}^{Q} &= W^{DQ} \mathbf{h}_{t}, \\    
[\mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} &= W^{UQ} \mathbf{c}_{t}^{Q}, \\ [\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \text{RoPE}({W^{QR}} \mathbf{c}_{t}^{Q}), \\    
\mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}], \\ 
\mathbf{c}_{t}^{KV} &= W^{DKV} \mathbf{h}_{t}, \\    
[\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} &= W^{UK} \mathbf{c}_{t}^{KV}, \\ 
\mathbf{k}_{t}^{R} &= \text{RoPE}({W^{KR}} \mathbf{h}_{t}), \\    
\mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\    
[\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} &= W^{UV} \mathbf{c}_{t}^{KV}, \\ 
\mathbf{o}_{t, i} &= \sum_{j=1}^{t} \text{Softmax}_j(\frac{\mathbf{q}_{t, i}^T \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}, \\    
\mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],
\end{align}
$$


Where

- $d$ be the embedding dimension
- $n_h$ be the number of attention heads
- $d_h$ be the dimension per head
- $\mathbf{h}_{t} \in \mathbb{R}^{d}$ be the attention input of **the** $t\text{-th}$ **token** at an attention layer
- $d_c^{\prime} (\ll d_h n_h)$ denotes the query compression dimension
- $\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_c^{\prime}}$ is the compressed latent vector for queries
- $W^{DQ} \in \mathbb{R}^{d_c^{\prime} \times d}, W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c^{\prime}}$ are the down-projection and up-projection matrices for queries, respectively.
- $d_c (\ll d_h n_h)$ denotes the KV compression dimension
- $\mathbf{c}_{t}^{KV} \in \mathbb{R}^{d_c}$ is the compressed **latent** **vector** for keys and values
- $W^{DKV} \in \mathbb{R}^{d_c \times d}$ is the **down-projection** matrix; and $W^{UK},W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ are the **up-projection** matrices for keys and values, respectively.
- The decoupled RoPE strategy uses **additional** multi-head queries $\mathbf{q}_{t, i}^{R} \in \mathbb{R}^{d_h^R}$ and a **shared** key $\mathbf{k}_{t}^{R} \in \mathbb{R}^{d_h^R}$ to carry RoPE, where $d_h^R$ denotes the per-head dimension of the decoupled queries and key.
- $W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c^{\prime}}$ and $W^{KR} \in \mathbb{R}^{d_h^R \times d}$ are matrices to **produce the decouples queries and key**, respectively.



For DeepSeek-V2

- $d_c$ is set to $4d_h$ and $d^R_h$ is set to $\frac{d_h}{2}$
- additional RMS Norm layers after the compressed latent vectors

Slightly different from DeepSeek-V2, DeepSeek-V2-Lite does not compress the queries.


## Shape

| **Formula**                                      | **Params**                                   | **Shape**                                            | **FLOPs**                                  | **Comments**                                                 |
| ------------------------------------------------ | -------------------------------------------- | ---------------------------------------------------- | ------------------------------------------ |--------------------------------------------------------------|
| $\mathbf{c}_{t}^{Q} = W^{DQ} \mathbf{h}_{t}$     | $W^{DQ} \in \mathbb{R}^{d_c^{\prime} \times d}$ | $[b, s, d] \times [d, d^{\prime}_c] = [b, s, d^{\prime}_c]$ | $2bs \cdot d \cdot d^{\prime}_c$         | $d \to d^{\prime}_c $                                        |
| $\mathbf{q}_{t}^{C} = W^{UQ}\mathbf{c}_{t}^{Q}$  | $W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c^{\prime}}$ | $[b, s, d^{\prime}_c] \times [d^{\prime}_c, d_h n_h] = [b, s, d_h n_h]$ | $2bs \cdot d^{\prime}_c \cdot d_h n_h$   | $d^{\prime}_c \to d_h n_h$                                   |
| $\mathbf{q}_{t}^{R} = \text{RoPE}({W^{QR}} \mathbf{c}_{t}^{Q})$ | $W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c^{\prime}}$ | $[b, s, d^{\prime}_c] \times [d^{\prime}_c, d^R_h n_h] = [b, s, d^R_h n_h]$ | $2bs \cdot d^{\prime}_c \cdot d^R_h n_h$ | $d^{\prime}_c \to d^R_hn_h$, produce the decouples queries   |
| $\mathbf{c}_{t}^{KV} = W^{DKV} \mathbf{h}_{t}$   | $W^{DKV} \in \mathbb{R}^{d_c \times d}$      | $[b, s, d] \times [d, d_c] = [b, s, d_c]$            | $2bs \cdot d \cdot d_c$                  | $d \to d_c$                                                  |
| $\mathbf{k}_{t}^{C} = W^{UK} \mathbf{c}_{t}^{KV}$ | $W^{UK} \in \mathbb{R}^{d_h n_h \times d_c}$ | $[b, s, d_c] \times [d_c, d_h n_h] = [b, s, d_h n_h]$ | $2bs \cdot d_c \cdot d_hn_h$             | $d_c \to d_hn_h$                                             |
| $\mathbf{k}_{t}^{R} = \text{RoPE}({W^{KR}} \mathbf{h}_{t})$ | $W^{KR} \in \mathbb{R}^{d_h^R \times d}$     | $[b, s, d] \times [d, d^R_h] = [b, s, d^R_h]$       | $2bs \cdot d \cdot d^R_h$                | $d \to d^R_h$, produce the decouples key, shared among heads |
| $\mathbf{v}_{t}^{C} = W^{UV} \mathbf{c}_{t}^{KV}$ | $W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ | $[b, s, d_c] \times [d_c, d_h n_h] = [b, s, d_h n_h]$ | $2bs \cdot d_c \cdot d_hn_h$             | $dc \to d_hn_h$                                              |
| $\mathbf{o}_{t} = \sum_{j=1}^{t} \text{Softmax}_j(\frac{\mathbf{q}_{t}^T \mathbf{k}_{j}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j}^{C}$ | 0                                            |                                                      |                                            | w/ FlashAttention                                            |
| $\mathbf{u}_{t} = W^{O} \mathbf{o}_{t}$          | $W^O \in \mathbb R^{d \times d_h n_h}$       | $[b, s, d_h n_h] \times [d_h n_h, d] = [b, s, d]$    | $2bs \cdot d_hn_h \cdot d$               | $d_hn_h \to d$                                               |