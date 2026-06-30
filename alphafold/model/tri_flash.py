"""Pallas/Triton flash attention for the AF2 Evoformer (MSA + triangle).

Inference (forward) only, no backward pass

Computes, per (batch n, head h):
  softmax_k( scale * Q_q . K_k + bias_{h,q,k}  if mask_{n,k} else -inf ) . V_k
Layout: q/k/v [N, H, S, D]; bias [H, Sq, Sk] (shared across N); kmask [N, Sk] bool.
"""
import functools, math
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

# jnp.dot is stable across jax versions, contrary to pl.dot/plgpu.dot
_dot = functools.partial(jnp.dot, preferred_element_type=jnp.float32)

NEG = -0.7 * float(np.finfo(np.float32).max)
LOG2E = math.log2(math.e)


def _kernel(q_ref, k_ref, v_ref, bias_ref, kmask_ref, o_ref, *,
            sm_scale, block_q, block_k, sq):
  # q_ref [block_q, D]; k_ref/v_ref [Sk, D]; bias_ref [block_q, Sk]; kmask_ref [Sk].
  # Sq/Sk are NOT padded to block multiples; boundary blocks are handled with
  # masked loads (offs < Sk) and a masked output store (offs_q < Sq) so no
  # jnp.pad copy of q/k/v is needed.
  D = q_ref.shape[-1]
  sk = k_ref.shape[0]
  q = q_ref[...]                                   # [block_q, D]
  m_i = jnp.full(block_q, -float('inf'), jnp.float32)
  l_i = jnp.zeros(block_q, jnp.float32)
  o = jnp.zeros((block_q, D), jnp.float32)

  def body(j, carry):
    o_prev, m_prev, l_prev = carry
    start = j * block_k
    kb = (start + jnp.arange(block_k)) < sk        # [block_k] in-bounds keys
    sl = pl.dslice(start, block_k)
    k = plgpu.load(k_ref.at[sl, :], mask=kb[:, None], other=0.0)   # [block_k, D]
    qk = _dot(q, k.T)                          # [block_q, block_k]
    bias = plgpu.load(bias_ref.at[:, sl], mask=kb[None, :], other=0.0)
    qk = (qk * sm_scale + bias) * LOG2E
    km = plgpu.load(kmask_ref.at[sl], mask=kb, other=False)  # OOB keys -> masked
    qk = jnp.where(km[None, :], qk, NEG)
    m_curr = jnp.max(qk, axis=-1)
    m_next = jnp.maximum(m_prev, m_curr)
    corr = jnp.exp2(m_prev - m_next)
    s = jnp.exp2(qk - m_next[:, None])
    l_next = corr * l_prev + s.sum(axis=-1)
    v = plgpu.load(v_ref.at[sl, :], mask=kb[:, None], other=0.0)   # [block_k, D]
    o_next = corr[:, None] * o_prev + _dot(s.astype(v.dtype), v)
    return o_next, m_next, l_next

  o, m_i, l_i = jax.lax.fori_loop(0, pl.cdiv(sk, block_k), body, (o, m_i, l_i))
  o = o / l_i[:, None]
  q_valid = (pl.program_id(0) * block_q + jnp.arange(block_q)) < sq
  plgpu.store(o_ref, o.astype(o_ref.dtype), mask=q_valid[:, None])


@functools.partial(jax.jit, static_argnames=(
    "sm_scale", "block_q", "block_k", "num_warps", "num_stages"))
def tri_flash(q, k, v, bias, kmask, *, sm_scale, block_q=64, block_k=64,
              num_warps=4, num_stages=2):
  # q/k/v [N,H,Sq,D]; bias [H,Sq,Sk]; kmask [N,Sk] bool. No padding: Sq/Sk need
  # not be block multiples; the kernel masks the partial q-block (output) and the
  # partial k-block (loads), so the big q/k/v tensors are never copied.
  N, H, Sq, D = q.shape
  Sk = k.shape[2]
  # grid (q_block, head, row): row N is innermost so the shared bias[h,q_block,:]
  # stays hot in L2 while sweeping rows (the bias is re-read for every row).
  grid = (pl.cdiv(Sq, block_q), H, N)
  bs_qo = pl.BlockSpec((None, None, block_q, D), lambda i, h, j: (j, h, i, 0))
  bs_kv = pl.BlockSpec((None, None, Sk, D),      lambda i, h, j: (j, h, 0, 0))
  bs_bias = pl.BlockSpec((None, block_q, Sk),    lambda i, h, j: (h, i, 0))
  bs_mask = pl.BlockSpec((None, Sk),             lambda i, h, j: (j, 0))
  return pl.pallas_call(
      functools.partial(_kernel, sm_scale=sm_scale, block_q=block_q,
                        block_k=block_k, sq=Sq),
      grid=grid,
      in_specs=[bs_qo, bs_kv, bs_kv, bs_bias, bs_mask],
      out_specs=bs_qo,
      out_shape=jax.ShapeDtypeStruct((N, H, Sq, D), q.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=num_warps, num_stages=num_stages),
      name="tri_flash_fwd",
  )(q, k, v, bias, kmask)


def pallas_attention(q, k, v, mask_bias, nonbatched_bias, scale):
  """AF2 attention via the Pallas flash kernel.

  q/k/v heads-major [b, h, S, c]; mask_bias additive [b,1,1,S_kv] (~-1e9 = invalid);
  nonbatched_bias [h,S_qo,S_kv] shared, or None. Returns [b, h, S_qo, c].
  """
  b, h, sq, c = q.shape
  sk = k.shape[2]
  # The kernel reuses q's channel dim for v (one BlockSpec), so key_dim must equal value_dim
  # True for AF2's attention
  assert q.shape[-1] == v.shape[-1], (
      f'tri_flash needs key_dim == value_dim, got {q.shape[-1]} vs {v.shape[-1]}')
  kmask = mask_bias[:, 0, 0, :] > -1e8                          # [b, S_kv] bool
  bias = (jnp.zeros((h, sq, sk), q.dtype)
          if nonbatched_bias is None else nonbatched_bias.astype(q.dtype))
  # 64x64/4w/2s is optimal across GB10/L40S/H100 (head_dim is only 32, so bigger
  # tiles waste the SM and are measurably slower).
  return tri_flash(q, k, v, bias, kmask, sm_scale=float(scale),
                   block_q=64, block_k=64).astype(q.dtype)


def ref_attn(q, k, v, bias, kmask, sm_scale):
  logits = jnp.einsum('nhqd,nhkd->nhqk', q, k) * sm_scale + bias[None]
  logits = jnp.where(kmask[:, None, None, :], logits, NEG)
  w = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
  return jnp.einsum('nhqk,nhkd->nhqd', w, v)


