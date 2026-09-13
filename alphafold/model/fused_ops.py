"""Pick the fused kernel (volta_attn, Pallas, or None for XLA) for each op."""
import functools


# Transitions aren't quadratic: chunk to a ~256 MiB intermediate, not 4 rows.
# Only with fused kernels, where the launch count was measured: on plain XLA the
# wider chunk makes it pick fusions that want more shared memory than RDNA has.
_TRANSITION_BUDGET_BYTES = 256 * 1024 * 1024


def transition_subbatch(global_config, shape, num_intermediate, dtype):
  """Subbatch that keeps a transition's intermediate near the budget."""
  import jax.numpy as jnp
  configured = global_config.subbatch_size
  if configured is None or not global_config.get('use_pallas', False):
    return configured
  per_row = num_intermediate * jnp.dtype(dtype).itemsize
  for dim in shape[1:-1]:
    per_row *= int(dim)
  if per_row <= 0:
    return configured
  return max(configured, min(shape[0], _TRANSITION_BUDGET_BYTES // per_row))


def _legacy_cc(global_config):
  """Give the compute capability if the CUDA kernels were asked for, else None."""
  if global_config.get('kernel_backend', 'pallas') != 'cuda_legacy':
    return None
  return global_config.get('compute_capability')


def _volta():
  from alphafold.model import volta_attn
  return volta_attn


def attention(global_config, dtype, key_dim, value_dim):
  """-> f(q, k, v, mask_bias, nonbatched_bias, scale) or None."""
  import jax.numpy as jnp
  if not global_config.get('use_pallas', False):
    return None
  cc = _legacy_cc(global_config)
  if dtype == jnp.float16 and cc is not None:
    va = _volta()
    # The kernels only handle a fixed set of head dims, and need key == value.
    if (va.available(cc) and key_dim == value_dim
        and va.supports(int(key_dim), cc)):
      return functools.partial(va.volta_attention, cc=cc)
  if (global_config.get('kernel_backend', 'pallas') == 'pallas'
      and dtype in (jnp.bfloat16, jnp.float16)):
    from alphafold.model.tri_flash import pallas_attention
    return pallas_attention
  return None


def attention_fused(global_config, config, act):
  """True if Attention(config) on `act` runs a fused kernel (no subbatching)."""
  key_dim = config.get('key_dim', int(act.shape[-1])) // config.num_head
  value_dim = config.get('value_dim', int(act.shape[-1])) // config.num_head
  return attention(global_config, act.dtype, key_dim, value_dim) is not None


def layer_norm(global_config, dtype):
  """-> f(x, scale, offset, *, eps) or None. Last-axis, scale+offset only."""
  import jax.numpy as jnp
  if not global_config.get('use_pallas', False):
    return None
  cc = _legacy_cc(global_config)
  if dtype == jnp.float16 and cc is not None:
    va = _volta()
    if va.ops_available(cc):
      return va.volta_layer_norm
  if (global_config.get('kernel_backend', 'pallas') == 'pallas'
      and dtype == jnp.bfloat16):
    from alphafold.model.tri_mul import pallas_layer_norm
    return pallas_layer_norm
  return None


def gated_dual_proj(global_config, dtype):
  """-> f(x, wp, bp, wg, bg, mask, *, split) or None."""
  import jax.numpy as jnp
  if not global_config.get('use_pallas', False):
    return None
  cc = _legacy_cc(global_config)
  if dtype == jnp.float16 and cc is not None:
    va = _volta()
    if va.ops_available(cc):
      return functools.partial(va.volta_gated_dual_proj, cc=cc)
  if (global_config.get('kernel_backend', 'pallas') == 'pallas'
      and dtype in (jnp.bfloat16, jnp.float16)):
    from alphafold.model.tri_mul import gated_dual_proj as gdp
    return gdp
  return None
