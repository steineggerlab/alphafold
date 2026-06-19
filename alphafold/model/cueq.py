"""cuEquivariance fused kernels for ``modules.Attention`` / TriangleMultiplication.

Attention layout, AF2 -> cuEq ``triangle_attention``:
  batch ``b`` -> ``N``; queries -> ``S_qo``; keys -> ``S_kv``;
  ``nonbatched_bias`` [h,q,k] -> ``bias`` [1,1,h,q,k]; additive mask -> bool ``mask``.
cuEq needs S_kv a multiple of 8 and head_dim divisible by 8, so we pad the
sequence axes and crop the queries back.
"""

import jax
import jax.numpy as jnp
import haiku as hk

# cuequivariance_jax is a lazy imported optional dependency
_AVAILABLE = None
def available():
  """True if cuequivariance_jax is importable."""
  global _AVAILABLE
  if _AVAILABLE is None:
    try:
      import cuequivariance_jax  # noqa: F401
      _AVAILABLE = True
    except Exception:
      _AVAILABLE = False
  return _AVAILABLE


def _pad8(n):
  return (n + 7) // 8 * 8


def attention(q, k, v, mask_bias, nonbatched_bias, scale):
  """Flash triangle-attention core (replaces score+softmax+value), AF2 layout.

  q/k/v [b, S, h, c] (q NOT pre-scaled, `scale` applied here); mask_bias additive
  [b,1,1,S_kv] (~-1e9 = invalid); nonbatched_bias [h,S_qo,S_kv] shared, or None.
  Returns [b, S_qo, h, c] with q's dtype.
  """
  import cuequivariance_jax as cuex

  b, sq, h, c = q.shape
  sk = k.shape[1]
  sqp, skp = _pad8(sq), _pad8(sk)
  dt = q.dtype

  def to_cueq(x, s, sp):              # [b, S, h, c] -> [1, b, h, Sp, c]
    x = jnp.transpose(x, (0, 2, 1, 3))[None]
    if sp != s:
      x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, sp - s), (0, 0)))
    return x

  cq = to_cueq(q, sq, sqp)
  ck = to_cueq(k, sk, skp)
  cv = to_cueq(v, sk, skp)

  # boolean key mask [1, b, 1, 1, S_kv], False on padded positions
  valid = (mask_bias > -1e8)                      # [b, 1, 1, sk]
  valid = valid[None]                             # [1, b, 1, 1, sk]
  if skp != sk:
    valid = jnp.pad(valid, ((0, 0), (0, 0), (0, 0), (0, 0), (0, skp - sk)))
  cmask = valid

  # additive bias [1, 1, h, S_qo, S_kv]
  if nonbatched_bias is None:
    cbias = jnp.zeros((1, 1, h, sqp, skp), dt)
  else:
    cbias = nonbatched_bias[None, None]           # [1, 1, h, sq, sk]
    cbias = jnp.pad(cbias, ((0, 0), (0, 0), (0, 0),
                            (0, sqp - sq), (0, skp - sk))).astype(dt)

  out = cuex.triangle_attention(cq, ck, cv, cbias, cmask, scale=float(scale))[0]
  out = out[:, :, :, :sq, :]                      # crop padded queries
  # [1, b, h, sq, c] -> [b, sq, h, c]
  return jnp.transpose(out[0], (0, 2, 1, 3)).astype(dt)


def triangle_multiply(act, mask, equation, num_intermediate):
  """Fused cuEq TriangleMultiplication for AF2's `_fused_triangle_multiplication`.

  Rebuilds AF2's fused Haiku params (left_norm_input, projection, gate,
  center_norm, output_projection, gating_linear) so the same checkpoint loads,
  then feeds them to ``triangle_multiplicative_update`` (output-first weights).

  act: pair [N_res, N_res, c_z]; mask: pair mask (1 = valid); equation
  'ikc,jkc->ijc' = outgoing else incoming; num_intermediate = c_i.
  Returns updated pair [N_res, N_res, c_z].
  """
  import cuequivariance_jax as cuex

  c_z = act.shape[-1]
  ci = num_intermediate
  dt = act.dtype

  def _ln(scope, dim):
    with hk.experimental.name_scope(scope):
      s = hk.get_parameter('scale', (dim,), dt, init=hk.initializers.Constant(1.))
      o = hk.get_parameter('offset', (dim,), dt, init=hk.initializers.Constant(0.))
    return s, o

  def _lin(scope, in_d, out_d):
    with hk.experimental.name_scope(scope):
      w = hk.get_parameter('weights', (in_d, out_d), dt,
                           init=hk.initializers.TruncatedNormal())
      b = hk.get_parameter('bias', (out_d,), dt, init=hk.initializers.Constant(0.))
    return w, b

  nin_w, nin_b = _ln('left_norm_input', c_z)
  p_w, p_b = _lin('projection', c_z, 2 * ci)
  g_w, g_b = _lin('gate', c_z, 2 * ci)
  nout_w, nout_b = _ln('center_norm', ci)
  po_w, po_b = _lin('output_projection', ci, c_z)
  go_w, go_b = _lin('gating_linear', c_z, c_z)

  direction = 'outgoing' if equation.replace(' ', '').startswith('ikc') else 'incoming'

  # AF2 fuses projections as [left | right] and picks direction via the einsum;
  # cuEq bakes it into the kernel (half 1 = a, half 2 = b).  For 'incoming' the
  # AF2 einsum maps a<->right, b<->left, so swap the two halves (proj + gate).
  if direction == 'incoming':
    def _swap(x):              # swap the two ci-blocks along the last axis
      a, b = jnp.split(x, 2, axis=-1)
      return jnp.concatenate([b, a], axis=-1)
    p_w, p_b = _swap(p_w), _swap(p_b)
    g_w, g_b = _swap(g_w), _swap(g_b)

  out = cuex.triangle_multiplicative_update(
      act, direction=direction, mask=mask,
      norm_in_weight=nin_w, norm_in_bias=nin_b,
      p_in_weight=p_w.T, p_in_bias=p_b,
      g_in_weight=g_w.T, g_in_bias=g_b,
      norm_out_weight=nout_w, norm_out_bias=nout_b,
      p_out_weight=po_w.T, p_out_bias=po_b,
      g_out_weight=go_w.T, g_out_bias=go_b)
  return out.astype(dt)
