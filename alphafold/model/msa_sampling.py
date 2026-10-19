"""AF2-multimer's per-recycle MSA sampling as its own jitted function: the model then takes fixed-shape features
(num_msa cluster rows, num_extra_msa extra rows) and compiles once per length; the sampler once per raw-depth bucket."""
import haiku as hk
import jax
import numpy as np

from alphafold.model import modules_multimer
from alphafold.model import prng

RAW = ('msa', 'msa_mask', 'deletion_matrix', 'bert_mask', 'cluster_bias_mask')
SAMPLED = ('msa', 'msa_mask', 'deletion_matrix', 'bert_mask', 'true_msa', 'cluster_profile', 'cluster_deletion_mean',
           'extra_msa', 'extra_msa_mask', 'extra_deletion_matrix')


def depth_bucket(depth, floor):
  """Raw rows the sampler compiles for: at least floor, the rows it samples, past it a power of two."""
  return floor if depth <= floor else 1 << (depth - 1).bit_length()


def pad_raw(feat, rows):
  """feat's raw MSA features with zero rows appended up to rows: sample_msa ranks them last and they weigh nothing."""
  out = {}
  for k in RAW:
    if k in feat:
      x = np.asarray(feat[k])
      out[k] = np.pad(x, [(0, rows - len(x))] + [(0, 0)] * (x.ndim - 1))
  return out


def _keys():
  # what AlphaFold, AlphaFoldIteration's ensemble step and EmbeddingsAndEvoformer draw from RunModel.apply's rng
  _, sub = prng.SafeKey(hk.next_rng_key()).split()
  _, sample_key, mask_key = sub.split(3)
  return sample_key.get(), mask_key.get()


def make_sampler(config):
  """-> jitted f(raw, apply_key): the features the model takes in place of raw for the recycle applied with apply_key.

  config is model.embeddings_and_evoformer; raw has at least num_msa + num_extra_msa rows (pad_raw).
  """
  keys = hk.transform(_keys).apply

  def sample(raw, apply_key):
    sample_key, mask_key = keys(None, apply_key)
    batch = dict(raw)
    batch['msa_profile'] = modules_multimer.make_msa_profile(batch)
    batch = modules_multimer.sample(batch, prng.SafeKey(sample_key), prng.SafeKey(mask_key), config)
    for k in ('extra_msa', 'extra_msa_mask', 'extra_deletion_matrix'):
      batch[k] = batch[k][:config.num_extra_msa]
    return {k: batch[k] for k in SAMPLED if k in batch}

  return jax.jit(sample)
