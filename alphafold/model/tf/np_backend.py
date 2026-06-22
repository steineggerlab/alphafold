"""NumPy backend that mimics a slice of the ``tensorflow.compat.v1`` API used by AF2

The model itself is pure JAX; TensorFlow was only used to run a handful of
array transforms on a feature dict on the CPU. This module provides drop-in
replacements for those ops backed by NumPy or Jax.

The random ops are bit-exact to TF's stateful Philox RNG.
The stochastic transforms reproduce TF's output for a given seed.
"""
import builtins
import contextlib
import functools
import types

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as _sparse
import tree

_slice = builtins.slice  # saved because this module defines a `slice` function

# dtypes
class _DType:
  """Wraps a NumPy dtype and adds the ``.min``/``.max`` attributes TF dtypes have."""

  def __init__(self, np_dtype):
    self.np = np.dtype(np_dtype)

  @property
  def _is_int(self):
    return np.issubdtype(self.np, np.integer)

  @property
  def min(self):
    return np.iinfo(self.np).min if self._is_int else np.finfo(self.np).min

  @property
  def max(self):
    return np.iinfo(self.np).max if self._is_int else np.finfo(self.np).max

  def __eq__(self, other):
    try:
      other_np = other.np if isinstance(other, _DType) else np.dtype(other)
    except TypeError:
      return NotImplemented
    return self.np == other_np

  def __hash__(self):
    return hash(self.np)

  def __repr__(self):
    return f"tf.{self.np.name}"


int32 = _DType(np.int32)
int64 = _DType(np.int64)
float32 = _DType(np.float32)
float64 = _DType(np.float64)
bool = _DType(np.bool_)      # noqa: A001 - mirrors tf.bool
string = _DType(np.object_)  # tf.string features are NumPy object arrays

dtypes = types.SimpleNamespace(DType=_DType)
Tensor = np.ndarray          # for isinstance checks / annotations

def _np_dtype(dtype):
  if dtype is None:
    return None
  if isinstance(dtype, _DType):
    return dtype.np
  return np.dtype(dtype)


def _shape_tuple(shape):
  if shape is None:
    return ()
  if isinstance(shape, (list, tuple)):
    return tuple(int(s) for s in shape)
  if isinstance(shape, np.ndarray):
    return tuple(int(s) for s in shape.tolist())
  return (int(shape),)


# array creation / inspection
def constant(value, dtype=None, shape=None, name=None):
  arr = np.array(value, dtype=_np_dtype(dtype))
  if shape is not None:
    arr = np.broadcast_to(arr, _shape_tuple(shape)).copy() \
      if arr.shape == () else np.reshape(arr, _shape_tuple(shape))
  return arr


def convert_to_tensor(value, dtype=None):
  return np.asarray(value, dtype=_np_dtype(dtype))


def ones(shape, dtype=float32):
  return np.ones(_shape_tuple(shape), dtype=_np_dtype(dtype))


def zeros(shape, dtype=float32):
  return np.zeros(_shape_tuple(shape), dtype=_np_dtype(dtype))


def ones_like(x):
  return np.ones_like(np.asarray(x))


def shape(x):
  return np.array(np.asarray(x).shape, dtype=np.int32)


def size(x):
  return int(np.asarray(x).size)


def range(start, limit=None, delta=1, dtype=None, name=None):  # noqa: A001
  if limit is None:
    start, limit = 0, start
  return np.arange(start, limit, delta, dtype=_np_dtype(dtype))


# dtype / shape manipulation
def cast(x, dtype):
  return np.asarray(x).astype(_np_dtype(dtype))


def reshape(tensor, shape, name=None):  # noqa: redefines `shape` arg intentionally
  return np.reshape(np.asarray(tensor), _shape_tuple(shape) if isinstance(
      shape, (list, tuple, np.ndarray)) else shape)


def expand_dims(x, axis):
  return np.expand_dims(np.asarray(x), axis)


def squeeze(x, axis=None):
  return np.squeeze(np.asarray(x), axis=axis)


def tile(x, multiples):
  return np.tile(np.asarray(x), _shape_tuple(multiples))


def concat(values, axis, name=None):
  # tf.concat takes its result dtype from the (first) tensor input and casts the
  # rest to match; np.concatenate would instead promote (e.g. a stray float64
  # constant -> float64). Cast to the first input's dtype to mirror TF.
  arrs = [np.asarray(v) for v in values]
  dtype = arrs[0].dtype
  return np.concatenate([a.astype(dtype, copy=False) for a in arrs], axis=axis)


def split(value, num_or_size_splits, axis=0):
  value = np.asarray(value)
  if isinstance(num_or_size_splits, (list, tuple, np.ndarray)):
    indices = np.cumsum([int(s) for s in num_or_size_splits])[:-1]
    return np.split(value, indices, axis=axis)
  return np.split(value, int(num_or_size_splits), axis=axis)


def pad(tensor, paddings, mode="CONSTANT", constant_values=0, name=None):
  return np.pad(np.asarray(tensor), paddings, mode="constant",
                constant_values=constant_values)


def slice(input_, begin, size, name=None):  # noqa: A001 - mirrors tf.slice
  input_ = np.asarray(input_)
  idx = []
  for b, s in zip(begin, size):
    b = int(b)
    idx.append(_slice(b, None) if int(s) == -1 else _slice(b, b + int(s)))
  return input_[tuple(idx)]


def gather(params, indices, axis=0, name=None):
  return np.take(np.asarray(params), np.asarray(indices), axis=axis)


def one_hot(indices, depth, axis=-1, dtype=float32, on_value=1.0, off_value=0.0):
  indices = np.asarray(indices)
  out = np.full(indices.shape + (depth,), off_value, dtype=_np_dtype(dtype))
  np.put_along_axis(out, indices[..., None], on_value, axis=-1)
  if axis not in (-1, out.ndim - 1):
    out = np.moveaxis(out, -1, axis)
  return out


# elementwise / reductions
def where(condition, x=None, y=None):
  if x is None and y is None:
    return np.argwhere(np.asarray(condition))
  return np.where(np.asarray(condition), x, y)


def clip_by_value(t, clip_value_min, clip_value_max):
  return np.clip(np.asarray(t), clip_value_min, clip_value_max)


def minimum(x, y):
  return np.minimum(x, y)


def maximum(x, y):
  return np.maximum(x, y)


def equal(x, y):
  return np.equal(x, y)


def logical_and(x, y):
  return np.logical_and(x, y)


def floor(x):
  return np.floor(x)


def log(x):
  return np.log(x)


def atan(x):
  return np.arctan(x)


def argmax(input, axis=None, output_type=int64, name=None):  # noqa: A002
  return np.argmax(np.asarray(input), axis=axis).astype(_np_dtype(output_type))


def argsort(values, axis=-1, direction="ASCENDING", name=None):
  out = np.argsort(np.asarray(values), axis=axis)
  return np.flip(out, axis=axis) if direction == "DESCENDING" else out


def sort(values, axis=-1, name=None):
  return np.sort(np.asarray(values), axis=axis)


def unique(x, name=None):
  # tf.unique returns (y, idx); the pipeline only uses [0]. Input is pre-sorted.
  y, idx = np.unique(np.asarray(x), return_inverse=True)
  return (y, idx)


def reduce_mean(input_tensor, axis=None, name=None):
  return np.mean(np.asarray(input_tensor), axis=axis)


def matmul(a, b, transpose_a=False, transpose_b=False, name=None):
  a = np.asarray(a)
  b = np.asarray(b)
  if transpose_a:
    a = np.swapaxes(a, -1, -2)
  if transpose_b:
    b = np.swapaxes(b, -1, -2)
  return np.matmul(a, b)


def tensordot(a, b, axes, name=None):
  return np.tensordot(np.asarray(a), np.asarray(b), axes)


def einsum(equation, *inputs, name=None):
  return np.einsum(equation, *[np.asarray(x) for x in inputs])


# math.* / sets.* / sparse.* namespaces
def _unsorted_segment_sum(data, segment_ids, num_segments, name=None):
  # Grouped row-sum via a sparse scatter-matmul: ~16x faster than np.add.at on
  # the big [extra_seq, num_res, 23] tensors, and exact (inputs are integer-
  # valued 0/1 masks/one-hots and integer deletion counts).
  data = np.asarray(data)
  num_segments = int(num_segments)
  n = data.shape[0]
  out_shape = (num_segments,) + data.shape[1:]
  if n == 0:
    return np.zeros(out_shape, data.dtype)
  scatter = _sparse.csr_matrix(
      (np.ones(n, data.dtype), (np.asarray(segment_ids), np.arange(n))),
      shape=(num_segments, n))
  return (scatter @ data.reshape(n, -1)).reshape(out_shape)


math = types.SimpleNamespace(
    minimum=minimum,
    maximum=maximum,
    unsorted_segment_sum=_unsorted_segment_sum,
)


def _sets_difference(a, b, name=None):
  # Inputs are [1, N] row sets; return the dense [1, M] difference (sorted).
  a = np.asarray(a).ravel()
  b = np.asarray(b).ravel()
  return np.setdiff1d(a, b)[None]


sets = types.SimpleNamespace(difference=_sets_difference)
# _sets_difference already returns a dense array; to_dense is the identity.
sparse = types.SimpleNamespace(to_dense=lambda sp, name=None: np.asarray(sp))


# bit-exact TensorFlow Philox. set_random_seed resets a per-"graph" op-seed counter, same as tf.Graph
# stateful ops without an explicit seed consume that counter in call order
from alphafold.model.tf import random_philox

_DEFAULT_GRAPH_SEED = 87654321
_MAXINT32 = 2 ** 31 - 1
_graph_seed = None
_auto_op_seed = 0
_ensemble_iter = 0
_fallback_rng = np.random.default_rng()


def set_random_seed(seed):
  # Reset all RNG state so each np_example_to_features
  global _graph_seed, _auto_op_seed, _ensemble_iter
  _graph_seed = None if seed is None else int(seed)
  _auto_op_seed = 0
  _ensemble_iter = 0


def _get_seed(op_seed=None):
  """Replicates random_seed.get_seed (graph mode): (graph_seed, op_seed)."""
  global _auto_op_seed
  if _graph_seed is not None:
    if op_seed is None:
      op_seed = _auto_op_seed
      _auto_op_seed += 1
    seeds = (_graph_seed % _MAXINT32, int(op_seed) % _MAXINT32)
  elif op_seed is not None:
    seeds = (_DEFAULT_GRAPH_SEED, int(op_seed) % _MAXINT32)
  else:
    return None # fully unseeded, nondeterministic fallback
  return (0, _MAXINT32) if seeds == (0, 0) else seeds


def _uniform(shape, minval=0, maxval=None, dtype=float32, seed=None, name=None):
  dt = _np_dtype(dtype) or np.float32
  gs = _get_seed(seed)
  if gs is None:
    size = _shape_tuple(shape)
    if np.issubdtype(dt, np.integer):
      return _fallback_rng.integers(int(minval), int(maxval), size=size, dtype=dt)
    return _fallback_rng.uniform(0.0 if minval is None else float(minval),
                                 1.0 if maxval is None else float(maxval),
                                 size=size).astype(dt)
  g, o = gs
  if np.issubdtype(dt, np.integer):
    return random_philox.uniform_int(g, o, _shape_tuple(shape), int(minval),
                                  int(maxval), iteration=_ensemble_iter).astype(dt)
  return random_philox.uniform_float(g, o, _shape_tuple(shape),
                                  iteration=_ensemble_iter).astype(dt)


def _stateless_uniform(shape, seed, minval=0, maxval=None, dtype=float32, name=None):
  # Stateless ops use a different seed->key scrambling than the stateful path.
  # In AF2 their outputs never reach the model (crop is a no-op: crop_size ==
  # num_res forces range 1 -> 0; template subsampling is disabled), so a plain
  # deterministic NumPy draw is sufficient here.
  rng = np.random.default_rng([int(s) for s in np.asarray(seed).astype(np.uint32).ravel()])
  dt = _np_dtype(dtype) or np.float32
  size = _shape_tuple(shape)
  if np.issubdtype(dt, np.integer):
    return rng.integers(int(minval), int(maxval), size=size, dtype=dt)
  return rng.uniform(0.0 if minval is None else float(minval),
                     1.0 if maxval is None else float(maxval), size=size).astype(dt)


def _categorical(logits, num_samples, dtype=int64, seed=None, name=None):
  gs = _get_seed(seed)
  logits = np.asarray(logits)
  if gs is None:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    cdf = np.cumsum(e, axis=-1)
    u = _fallback_rng.random((logits.shape[0], num_samples)) * cdf[:, -1:]
    out = np.stack([(cdf > u[:, j][:, None]).argmax(-1) for j in range(num_samples)], axis=1)
    return out.astype(_np_dtype(dtype))
  g, o = gs
  return random_philox.categorical(g, o, logits, num_samples,
                                iteration=_ensemble_iter).astype(_np_dtype(dtype))


def random_shuffle(value, seed=None, name=None):
  gs = _get_seed(seed)
  if gs is None:
    return _fallback_rng.permutation(np.asarray(value))
  g, o = gs
  return random_philox.random_shuffle(g, o, value, iteration=_ensemble_iter)


random = types.SimpleNamespace(
    uniform=_uniform,
    stateless_uniform=_stateless_uniform,
    categorical=_categorical,
)


# graph / session become no-ops
class Graph:
  def as_default(self):
    return contextlib.nullcontext()

  def finalize(self):
    pass


class Session:
  def __init__(self, graph=None, config=None):
    pass

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    return False

  def run(self, fetches, feed_dict=None):
    return fetches


def device(name):
  return contextlib.nullcontext()


def control_dependencies(control_inputs):
  return contextlib.nullcontext()


def disable_v2_behavior():
  pass


def assert_equal(x, y, data=None, summarize=None, message=None, name=None):
  assert int(np.asarray(x)) == int(np.asarray(y)), message or f"{x} != {y}"


def assert_greater(x, y, data=None, summarize=None, message=None, name=None):
  assert np.all(np.asarray(x) > np.asarray(y)), message or f"{x} !> {y}"


def map_fn(fn, elems, fn_output_signature=None, dtype=None,
           parallel_iterations=None, name=None, **kwargs):
  # tf.map_fn builds the body once: its random ops keep fixed op-seeds across
  # iterations and advance their Philox counter per call. Replicate that by
  # resetting the op-seed counter each iteration and signalling the iteration index 
  global _auto_op_seed, _ensemble_iter
  saved = _auto_op_seed
  results = []
  for k, e in enumerate(np.asarray(elems)):
    _auto_op_seed = 0
    _ensemble_iter = k
    results.append(fn(e))
  _ensemble_iter = 0
  _auto_op_seed = saved
  return tree.map_structure(lambda *xs: np.stack(xs), *results)


TensorSpec = types.SimpleNamespace(from_tensor=lambda x, name=None: None)


# tf.train.* / tf.io.*
class _Example:
  """Placeholder for `tf.train.Example`"""


class _Unused:
  def __init__(self, *args, **kwargs):
    raise NotImplementedError()


def _tfrecord_stub(*args, **kwargs):
  raise NotImplementedError()


train = types.SimpleNamespace(Example=_Example, Feature=_Unused, FloatList=_Unused)
io = types.SimpleNamespace(
    parse_single_example=_tfrecord_stub,
    FixedLenSequenceFeature=_tfrecord_stub,
)

import sys as _sys  # noqa: E402
compat = types.SimpleNamespace(v1=_sys.modules[__name__])


# Fast Jax MSA-clustering / profile transforms
# The [seqs, num_res, 23] one-hots are never materialized in numpy.
# Clustering (full pre-crop extra, up to ~60k) runs through fused, threaded Jax
# jits, on CPU for small/typical MSAs and automatically on the GPU for large ones
_GPU_THRESHOLD = 2_000_000  # num_extra*num_res; below this CPU is faster
_dev = {}


def _gpu_dev():
  if 'gpu' not in _dev:
    try:
      _dev['gpu'] = jax.devices('gpu')[0]
    except RuntimeError:
      _dev['gpu'] = None    # no GPU visible -> always CPU
  return _dev['gpu']


def _cluster_device(num_extra, num_res):
  g = _gpu_dev()
  if g is not None and num_extra * num_res >= _GPU_THRESHOLD:
    return g
  return _dev.setdefault('cpu', jax.local_devices(backend='cpu')[0])


def _to_dev(dev, *arrays):
  return [jax.device_put(np.asarray(a), dev) for a in arrays]


# Disable XLA's Triton-GEMM autotuning for the clustering jits.
# On GPU it spends ~14s for autotuning Triton kernels;  ~150ms with it off and same runtime.
_COPTS = {'xla_gpu_enable_triton_gemm': False}


@functools.partial(jax.jit, compiler_options=_COPTS)
def _jit_cluster_assign(msa, msa_mask, extra_msa, extra_mask):
  # Agreement = count of residues where an extra sequence matches a cluster
  # centre (a weighted Hamming similarity); only the 21 real-aa classes (0..20)
  # contribute (gap/X/MASK carry weight 0, so the 21-wide one-hot zeroes them).
  sample = (msa_mask[:, :, None].astype(jnp.int8)
            * jax.nn.one_hot(msa, 21, dtype=jnp.int8)).reshape(msa.shape[0], -1)
  extra_oh = (extra_mask[:, :, None].astype(jnp.int8)
              * jax.nn.one_hot(extra_msa, 21, dtype=jnp.int8)).reshape(
                  extra_msa.shape[0], -1)
  agreement = jax.lax.dot_general(extra_oh, sample, (((1,), (1,)), ((), ())),
                                  preferred_element_type=jnp.int32)
  return jnp.argmax(agreement, axis=1).astype(jnp.int32)


@functools.partial(jax.jit, compiler_options=_COPTS)
def _jit_cluster_sums(assign, msa, extra_msa, extra_mask, extra_deletion, deletion):
  num_seq, num_res = msa.shape
  counts = jax.ops.segment_sum(extra_mask, assign, num_segments=num_seq)
  # msa_sum via a direct scatter-add into (num_seq, num_res, 23), instead of
  # one-hotting the full extra MSA (a [num_extra, num_res, 23] tensor — 1.4GB at
  # 60k) and segment-summing it. ~3x faster + far less memory. extra_mask is 0/1
  # so the float accumulation is integer-exact (order-independent).
  msa_sum = jnp.zeros((num_seq, num_res, 23), jnp.float32).at[
      assign[:, None], jnp.arange(num_res)[None, :], extra_msa].add(extra_mask)
  msa_sum = msa_sum + jax.nn.one_hot(msa, 23)
  del_sum = (jax.ops.segment_sum(extra_mask * extra_deletion, assign,
                                 num_segments=num_seq) + deletion)
  return counts, msa_sum, del_sum


def fast_nearest_neighbor_clusters(protein, gap_agreement_weight=0.):
  extra = np.asarray(protein['extra_msa'])
  if extra.shape[0] == 0:
    protein['extra_cluster_assignment'] = np.zeros([0], dtype=np.int32)
    return protein
  dev = _cluster_device(extra.shape[0], extra.shape[1])
  args = _to_dev(dev, protein['msa'], protein['msa_mask'], protein['extra_msa'],
                 protein['extra_msa_mask'])
  protein['extra_cluster_assignment'] = np.asarray(_jit_cluster_assign(*args))
  return protein


def fast_randomly_replace_msa_with_unknown(protein):
  # consume the two op-seeds the two original tf.random.uniform calls
  # to keep downstream RNG aligned
  random.uniform([1])
  random.uniform([1])
  return protein


def fast_summarize_clusters(protein):
  extra = np.asarray(protein['extra_msa'])
  dev = _cluster_device(extra.shape[0], extra.shape[1])  # same device as assign
  counts, msa_sum, del_sum = _jit_cluster_sums(*_to_dev(dev,
      protein['extra_cluster_assignment'], protein['msa'], protein['extra_msa'],
      protein['extra_msa_mask'], protein['extra_deletion_matrix'],
      protein['deletion_matrix']))
  counts, msa_sum, del_sum = np.asarray(counts), np.asarray(msa_sum), np.asarray(del_sum)
  # float arithmetic stays in numpy
  mask_counts = 1e-6 + np.asarray(protein['msa_mask']) + counts
  protein['cluster_profile'] = msa_sum / mask_counts[:, :, None]
  protein['cluster_deletion_mean'] = del_sum / mask_counts
  return protein


def fast_make_hhblits_profile(protein):
  msa = np.asarray(protein['msa'])
  num_seq, num_res = msa.shape
  counts = np.bincount((np.arange(num_res) * 22 + msa).ravel(),
                       minlength=num_res * 22).reshape(num_res, 22)
  protein['hhblits_profile'] = counts.astype(np.float32) / np.float32(num_seq)
  return protein
