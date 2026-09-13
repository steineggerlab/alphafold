"""float16 helpers: sentinels that fit in fp16, and fp32 accumulation."""
import contextlib

import haiku as hk
import jax.numpy as jnp

__all__ = ['big_neg', 'logit_clip', 'mask_to_bias', 'half_dtype', 'half_creator',
           'half_getter', 'half_context', 'wide_einsum', 'to_half_like']

_HALF_DTYPES = (jnp.bfloat16, jnp.float16)


def big_neg(dtype):
  """Additive mask value: -1e9, or -1e4 in float16."""
  if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
    return -1e4
  return -1e9


def logit_clip(dtype):
  """Symmetric logit clip bound that is finite in `dtype`."""
  if jnp.dtype(dtype) == jnp.dtype(jnp.float16):
    return 1e4
  return 1e8


def mask_to_bias(mask, dtype=None):
  """AF2's `1e9 * (mask - 1.)` bias, with big_neg(dtype) for masked entries."""
  dtype = mask.dtype if dtype is None else dtype
  return (big_neg(dtype) * (1. - mask.astype(jnp.float32))).astype(dtype)


def half_dtype(global_config):
  """Trunk dtype: float32 unless global_config.bfloat16, then half_dtype."""
  if not global_config.bfloat16:
    return jnp.float32
  name = global_config.get('half_dtype', 'bfloat16')
  if name in ('float16', 'fp16', 'f16'):
    return jnp.float16
  if name in ('bfloat16', 'bf16'):
    return jnp.bfloat16
  raise ValueError(f'unknown half_dtype {name!r}')


def half_creator(next_creator, shape, dtype, init, context):
  """Creates float32 variables when a half dtype is requested."""
  if context.original_dtype in _HALF_DTYPES:
    dtype = jnp.float32
  return next_creator(shape, dtype, init)


def half_getter(next_getter, value, context):
  """Casts float32 params down to whichever half dtype was requested."""
  if context.original_dtype in _HALF_DTYPES:
    assert value.dtype == jnp.float32
    value = value.astype(context.original_dtype)
  return next_getter(value)


@contextlib.contextmanager
def half_context():
  """Param storage stays fp32; reads are cast to the requested half dtype."""
  with hk.custom_creator(half_creator), hk.custom_getter(half_getter):
    yield


def wide_einsum(equation, a, b, **kwargs):
  """einsum with an fp32 accumulator for float16 operands."""
  if jnp.dtype(a.dtype) == jnp.dtype(jnp.float16):
    kwargs.setdefault('preferred_element_type', jnp.float32)
  return jnp.einsum(equation, a, b, **kwargs)


def to_half_like(x, ref):
  """Cast back to `ref`'s dtype if `wide_einsum` widened the accumulation."""
  if jnp.dtype(ref.dtype) == jnp.dtype(jnp.float16):
    return x.astype(ref.dtype)
  return x
