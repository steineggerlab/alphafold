"""Pallas/Triton fused gated dual projection for AF2 triangle-multiplication.

Inference (forward) only, no backward pass

Computes the masked, sigmoid-gated dual projection in one kernel so the [M, 2c]
projection/gate intermediates are never materialized. The triangle matmul itself
stays a native XLA batched GEMM.

  out[m, p] = mask[m] * (x[m]·wp[:,p] + bp[p]) * sigmoid(x[m]·wg[:,p] + bg[p])
where x = layer-normed pair [M=N*N, K=c_z], wp/wg [K, P=2*c_i].
"""
import functools
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu


def _gdp_kernel(x_ref, wp_ref, bp_ref, wg_ref, bg_ref, mask_ref, o_ref):
  x = x_ref[...]                                      # [BM, K] bf16
  proj = plgpu.dot(x, wp_ref[...]) + bp_ref[...][None, :].astype(jnp.float32)
  gate = plgpu.dot(x, wg_ref[...]) + bg_ref[...][None, :].astype(jnp.float32)
  o = mask_ref[...][:, None].astype(jnp.float32) * proj * jax.nn.sigmoid(gate)
  o_ref[...] = o.astype(o_ref.dtype)


def _half(x, wp, bp, wg, bg, mask):
  proj = plgpu.dot(x, wp) + bp[None, :].astype(jnp.float32)
  gate = plgpu.dot(x, wg) + bg[None, :].astype(jnp.float32)
  return mask[:, None].astype(jnp.float32) * proj * jax.nn.sigmoid(gate)


def _gdp_kernel_split(x_ref, wpl_ref, bpl_ref, wgl_ref, bgl_ref,
                      wpr_ref, bpr_ref, wgr_ref, bgr_ref, mask_ref,
                      ol_ref, or_ref):
  # Computes the left [BM, ci] and right [BM, ci] gated projections as two
  # separate dots (weights pre-split on the host) and writes them to two
  # contiguous outputs -- so the triangle einsum consumes them directly with no
  # post-kernel strided slice/copy of the combined [N,N,2ci] tensor.
  x = x_ref[...]
  m = mask_ref[...]
  ol_ref[...] = _half(x, wpl_ref[...], bpl_ref[...], wgl_ref[...], bgl_ref[...],
                      m).astype(ol_ref.dtype)
  or_ref[...] = _half(x, wpr_ref[...], bpr_ref[...], wgr_ref[...], bgr_ref[...],
                      m).astype(or_ref.dtype)


@functools.partial(jax.jit, static_argnames=("block_m", "split"))
def gated_dual_proj(x, wp, bp, wg, bg, mask, *, block_m=64, split=False):
  # x [M,K]; wp/wg [K,P]; bp/bg [P]; mask [M].
  # split=False -> gated [M,P]; split=True -> (left [M,P/2], right [M,P/2]),
  # the two halves written contiguously so the caller needs no slice.
  M, K = x.shape
  P = wp.shape[1]
  Mp = -(-M // block_m) * block_m
  x = jnp.pad(x, [(0, Mp - M), (0, 0)])
  mask = jnp.pad(mask, [(0, Mp - M)])
  in_specs = [
      pl.BlockSpec((block_m, K), lambda i: (i, 0)),
      pl.BlockSpec((K, P), lambda i: (0, 0)),
      pl.BlockSpec((P,), lambda i: (0,)),
      pl.BlockSpec((K, P), lambda i: (0, 0)),
      pl.BlockSpec((P,), lambda i: (0,)),
      pl.BlockSpec((block_m,), lambda i: (i,)),
  ]
  cp = plgpu.CompilerParams(num_warps=4, num_stages=1)
  if split:
    ci = P // 2
    wpl, wpr = wp[:, :ci], wp[:, ci:]
    wgl, wgr = wg[:, :ci], wg[:, ci:]
    bpl, bpr = bp[:ci], bp[ci:]
    bgl, bgr = bg[:ci], bg[ci:]
    wspec = pl.BlockSpec((K, ci), lambda i: (0, 0))
    bspec = pl.BlockSpec((ci,), lambda i: (0,))
    split_in = [in_specs[0], wspec, bspec, wspec, bspec, wspec, bspec,
                wspec, bspec, in_specs[5]]
    bs = pl.BlockSpec((block_m, ci), lambda i: (i, 0))
    left, right = pl.pallas_call(
        _gdp_kernel_split,
        grid=(Mp // block_m,), in_specs=split_in, out_specs=[bs, bs],
        out_shape=[jax.ShapeDtypeStruct((Mp, ci), x.dtype),
                   jax.ShapeDtypeStruct((Mp, ci), x.dtype)],
        compiler_params=cp, name="gated_dual_proj_split",
    )(x, wpl, bpl, wgl, bgl, wpr, bpr, wgr, bgr, mask)
    return left[:M], right[:M]
  out = pl.pallas_call(
      _gdp_kernel,
      grid=(Mp // block_m,), in_specs=in_specs,
      out_specs=pl.BlockSpec((block_m, P), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((Mp, P), x.dtype),
      compiler_params=cp, name="gated_dual_proj",
  )(x, wp, bp, wg, bg, mask)
  return out[:M]


def ref_gdp(x, wp, bp, wg, bg, mask):
  proj = x @ wp + bp
  gate = x @ wg + bg
  return (mask[:, None] * proj * jax.nn.sigmoid(gate))


# Fused LayerNorm: bf16 in/out, fp32 internal. One kernel replaces the
# bf16->fp32 upcast + layernorm + fp32->bf16 downcast (kills the convert ops
# and never materializes the fp32 tensor), while keeping fp32 accuracy.
def _ln_kernel(x_ref, scale_ref, offset_ref, o_ref, *, eps, m, block_m):
  valid = (pl.program_id(0) * block_m + jnp.arange(block_m)) < m   # [block_m]
  x = plgpu.load(x_ref, mask=valid[:, None], other=0.0).astype(jnp.float32)
  mean = jnp.mean(x, axis=-1, keepdims=True)
  d = x - mean
  var = jnp.mean(d * d, axis=-1, keepdims=True)
  y = d * jax.lax.rsqrt(var + eps)
  y = y * scale_ref[...][None, :].astype(jnp.float32) + \
      offset_ref[...][None, :].astype(jnp.float32)
  plgpu.store(o_ref, y.astype(o_ref.dtype), mask=valid[:, None])


@functools.partial(jax.jit, static_argnames=("eps", "block_m"))
def pallas_layer_norm(x, scale, offset, *, eps=1e-5, block_m=64):
  # x [..., c]; normalize over last axis; scale/offset [c]. Returns x's shape/dtype.
  c = x.shape[-1]
  xf = x.reshape(-1, c)
  m = xf.shape[0]
  out = pl.pallas_call(
      functools.partial(_ln_kernel, eps=eps, m=m, block_m=block_m),
      grid=(pl.cdiv(m, block_m),),
      in_specs=[pl.BlockSpec((block_m, c), lambda i: (i, 0)),
                pl.BlockSpec((c,), lambda i: (0,)),
                pl.BlockSpec((c,), lambda i: (0,))],
      out_specs=pl.BlockSpec((block_m, c), lambda i: (i, 0)),
      out_shape=jax.ShapeDtypeStruct((m, c), x.dtype),
      compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
      name="pallas_layer_norm",
  )(xf, scale, offset)
  return out.reshape(x.shape)


def ref_layer_norm(x, scale, offset, eps=1e-5):
  xf = x.astype(jnp.float32)
  mean = jnp.mean(xf, -1, keepdims=True)
  var = jnp.mean((xf - mean) ** 2, -1, keepdims=True)
  y = (xf - mean) * jax.lax.rsqrt(var + eps) * scale + offset
  return y.astype(x.dtype)


if __name__ == "__main__":
  import time
  key = jax.random.PRNGKey(0)
  M, K, P = 1000 * 1000, 128, 256
  x = (jax.random.normal(key, (M, K)) * 0.3).astype(jnp.bfloat16)
  wp = (jax.random.normal(key, (K, P)) * 0.1).astype(jnp.bfloat16)
  wg = (jax.random.normal(key, (K, P)) * 0.1).astype(jnp.bfloat16)
  bp = (jax.random.normal(key, (P,)) * 0.1).astype(jnp.bfloat16)
  bg = (jax.random.normal(key, (P,)) * 0.1).astype(jnp.bfloat16)
  mask = (jax.random.uniform(key, (M,)) > 0.1).astype(jnp.bfloat16)
  o = gated_dual_proj(x, wp, bp, wg, bg, mask)
  r = ref_gdp(x.astype(jnp.float32), wp.astype(jnp.float32), bp.astype(jnp.float32),
              wg.astype(jnp.float32), bg.astype(jnp.float32), mask.astype(jnp.float32))
  err = float(jnp.max(jnp.abs(o.astype(jnp.float32) - r)))
  print(f"gated_dual_proj bf16 err={err:.3e}")
  f = jax.jit(gated_dual_proj)
  o = f(x, wp, bp, wg, bg, mask); o.block_until_ready()
  t = time.time()
  for _ in range(50): o = f(x, wp, bp, wg, bg, mask)
  o.block_until_ready()
  print(f"per-call {(time.time()-t)/50*1000:.2f} ms")

