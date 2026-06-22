"""Bit-exact  reimplementation of the TensorFlow stateful RNG (Philox-4x32-10)

Seed model the per-op seed is  a counter 0,1,2,... incrementing
per stateful random op in graph-construction order

Original ``tf.random.categorical`` with num_samples=1 is thread-count-dependent in
TF. We reproduce the deterministic *single-threaded* TF result
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

# Philox-4x32-10 constants
_M_A = np.uint32(0xD2511F53)
_M_B = np.uint32(0xCD9E8D57)
_W_A = np.uint32(0x9E3779B9)
_W_B = np.uint32(0xBB67AE85)
_U32 = np.uint32
_MASK32 = 0xFFFFFFFF


def _mul_hi_lo(a, b):
  """MultiplyHighLow: returns (low32, high32) of a*b for uint32 arrays a and scalar b."""
  prod = a.astype(np.uint64) * np.uint64(int(b))
  return (prod & np.uint64(_MASK32)).astype(np.uint32), (prod >> np.uint64(32)).astype(np.uint32)


def _philox_blocks(graph_seed, op_seed, nblocks, start_block=0):
  """Return an (nblocks, 4) uint32 array: Philox output for counters
  start_block..start_block+nblocks-1.

  key=(graph_seed_lo, graph_seed_hi); counter=(block, 0, op_seed_lo, op_seed_hi).
  ``start_block`` offsets the 128-bit counter (used to replicate the counter
  advance of a stateful op re-executed inside tf.map_fn).
  """
  gs = int(graph_seed) & 0xFFFFFFFFFFFFFFFF
  os_ = int(op_seed) & 0xFFFFFFFFFFFFFFFF
  key0 = _U32(gs & _MASK32)
  key1 = _U32((gs >> 32) & _MASK32)
  idx = np.arange(nblocks, dtype=np.uint64) + np.uint64(int(start_block))
  c0 = (idx & np.uint64(_MASK32)).astype(np.uint32)        # counter[0]
  c1 = (idx >> np.uint64(32)).astype(np.uint32)            # counter[1] (carry)
  c2 = np.full(nblocks, os_ & _MASK32, dtype=np.uint32)
  c3 = np.full(nblocks, (os_ >> 32) & _MASK32, dtype=np.uint32)
  k0 = np.full(nblocks, key0, dtype=np.uint32)
  k1 = np.full(nblocks, key1, dtype=np.uint32)
  with np.errstate(over="ignore"):
    for r in range(10):
      lo0, hi0 = _mul_hi_lo(c0, _M_A)                       # counter[0] * M_A
      lo1, hi1 = _mul_hi_lo(c2, _M_B)                       # counter[2] * M_B
      c0, c1, c2, c3 = (hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0)
      if r < 9:                                             # RaiseKey after first 9 rounds
        k0 = k0 + _W_A
        k1 = k1 + _W_B
  return np.stack([c0, c1, c2, c3], axis=1)


def _u32_stream(graph_seed, op_seed, n, start_block=0):
  """`n` uint32 of the block-major Philox stream (FillPhiloxRandom layout),
  beginning at 128-bit counter `start_block`."""
  if n <= 0:
    return np.zeros(0, dtype=np.uint32)
  nblocks = (n + 3) // 4
  return _philox_blocks(graph_seed, op_seed, nblocks, start_block).reshape(-1)[:n]


def _u32_to_float(u):
  """Uint32ToFloat: 23 low bits -> [1,2) -> [0,1) float32 (random_distributions_utils.h:33)."""
  man = u & np.uint32(0x7FFFFF)
  bits = np.uint32(0x3F800000) | man
  return bits.view(np.float32) - np.float32(1.0)


def _u64_to_double(x0, x1):
  """Uint64ToDouble: 52 mantissa bits -> [1,2) -> [0,1) double (random_distributions_utils.h:50)."""
  mhi = (x0 & np.uint32(0xFFFFF)).astype(np.uint64)
  mlo = x1.astype(np.uint64)
  man = (mhi << np.uint64(32)) | mlo
  val = (np.uint64(1023) << np.uint64(52)) | man
  return val.view(np.float64) - 1.0


# Distributions / ops
def uniform_float(graph_seed, op_seed, shape, iteration=0):
  shape = tuple(int(s) for s in shape)
  n = int(np.prod(shape)) if shape else 1
  start = iteration * n * 256                              # ReserveRandomOutputs(n, 256)
  out = _u32_to_float(_u32_stream(graph_seed, op_seed, n, start))
  return out.reshape(shape)


def uniform_int(graph_seed, op_seed, shape, minval, maxval, iteration=0):
  """UniformDistribution<int32>: lo + (sample % (hi-lo)), modulo (random_distributions.h:183)."""
  shape = tuple(int(s) for s in shape)
  n = int(np.prod(shape)) if shape else 1
  start = iteration * n * 256                              # ReserveRandomOutputs(n, 256)
  s = _u32_stream(graph_seed, op_seed, n, start)
  lo_u = _U32(int(minval) & _MASK32)                       # two's-complement bit pattern
  hi_u = _U32(int(maxval) & _MASK32)
  with np.errstate(over="ignore"):
    rng = hi_u - lo_u                                      # == range_ (uint32 wrap)
    assert rng != 0, "uniform_int requires minval != maxval (range_ == 0)"
    vals = (lo_u + (s % rng)).astype(np.uint32)            # SignedAdd via uint32 wrap
  return vals.view(np.int32).reshape(shape)


def random_shuffle(graph_seed, op_seed, value, iteration=0):
  """Fisher-Yates for i in [0,n-1): swap(i, i + single()%(n-i)).

  The swap offsets s[i] % (n-i) are computed vectorised; the sequential swaps
  then run on a plain Python index list (numpy scalar get/set in the loop is
  ~4x slower). Permuting an index array and gathering once is identical to
  swapping the rows in place, so it stays bit-exact and rank/dtype-agnostic.
  """
  value = np.asarray(value)
  n = value.shape[0]
  if n <= 1:
    return value.copy()
  start = iteration * ((n - 1 + 3) // 4)                    # ReserveSamples32(n-1) -> blocks
  s = _u32_stream(graph_seed, op_seed, n - 1, start)        # single() per step
  divisors = (n - np.arange(n - 1, dtype=np.int64)).astype(np.uint32)
  offsets = (s % divisors).tolist()                         # s[i] % (n-i), vectorised
  idx = list(range(n))
  for i, off in enumerate(offsets):
    j = i + off
    idx[i], idx[j] = idx[j], idx[i]
  return value[idx]


@functools.partial(jax.jit, backend='cpu')
def _categorical_inner(logits, doubles):
  """Inverse-CDF sampling: out[r,j] = upper_bound(cdf_r, RandDouble*running_total).

  Run as a fused, threaded jit-CPU subgraph (the exp/cumsum/argmax over the whole
  [rows, classes] tensor is the bulk of the categorical cost). float64 throughout;
  the Philox stream that produces ``doubles`` stays in NumPy (bit-exact). NOTE:
  XLA's exp/cumsum can differ from NumPy's by ~1 ULP, so in principle a sample can
  flip when ``to_find`` lands within 1 ULP of a cdf step (~3e-10/elt; 0 observed
  over 130k rows). This is the one spot NOT guaranteed bit-identical to NumPy/TF.
  """
  e = jnp.exp(logits - logits.max(axis=1, keepdims=True))
  cdf = jnp.cumsum(e, axis=1)                               # running_total cumulative
  to_find = doubles * cdf[:, -1:]                           # (rows, samples)
  # first class with cdf > to_find = upper_bound(cdf), per (row, sample)
  return (cdf[:, None, :] > to_find[:, :, None]).argmax(axis=2).astype(jnp.int32)


def categorical(graph_seed, op_seed, logits, num_samples, iteration=0):
  """Deterministic tf.random.categorical

  Per (row, sample): to_find = RandDouble() * running_total; out = upper_bound(cdf).
  RandDouble consumes 2 uint32 continuously in row-major order. The op reserves
  (num_samples+3)//4 blocks per row, so a map_fn re-execution advances by
  num_rows*((num_samples+3)//4) blocks.
  """
  logits = np.asarray(logits, dtype=np.float64)
  num_rows, num_classes = logits.shape
  num_samples_ceil_4 = (num_samples + 3) // 4 * 4 * 2
  start = iteration * num_rows * num_samples_ceil_4 * 256
  s = _u32_stream(graph_seed, op_seed, num_rows * num_samples * 2, start)
  doubles = _u64_to_double(s[0::2], s[1::2]).reshape(num_rows, num_samples)
  return np.asarray(_categorical_inner(logits, doubles))
