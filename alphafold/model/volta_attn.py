"""sm_70/75 CUDA kernels (colabfold-legacy-kernels) used instead of Pallas."""
import ctypes
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

# Head dims each kernel instantiates: mma.sync (sm_75+) and wmma (sm_70).
_HEAD_DIMS_MMA = (8, 16, 32, 64)
_HEAD_DIMS_WMMA = (16, 32, 64)

_LOADED = {}


def _package_path(kernel, cc):
    """Give the library path from colabfold_legacy_kernels, or None."""
    try:
        import colabfold_legacy_kernels as clk
    except ImportError:
        return None
    try:
        return clk.library_path(kernel, cc)
    except (FileNotFoundError, KeyError):
        return None


def _load(kernel, cc, symbols):
    """dlopen the library for one kernel and register its FFI targets."""
    key = (kernel, cc, tuple(symbols))
    if key in _LOADED:
        return _LOADED[key]
    path = _package_path(kernel, cc)
    if not path or not os.path.exists(path):
        _LOADED[key] = False
        return False
    lib = ctypes.cdll.LoadLibrary(path)
    for sym in symbols:
        jax.ffi.register_ffi_target(
            sym, jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
    _LOADED[key] = True
    return True


def _attn_symbol(cc):
    """sm_75+ gets the CUTLASS kernel; sm_70 the wmma one."""
    return "VoltaMma" if cc >= 75 else "VoltaWmma"


def available(cc):
    """True if the attention library for this device loads."""
    return _load("attention", cc, (_attn_symbol(cc),))


def _native_dims(cc):
    return _HEAD_DIMS_MMA if cc >= 75 else _HEAD_DIMS_WMMA


def _pad_target(head_dim, cc):
    """Smallest supported head dim >= head_dim, or None."""
    for d in sorted(_native_dims(cc)):
        if d >= head_dim:
            return d
    return None


def supports(head_dim, cc):
    if head_dim in _native_dims(cc):
        return True
    # wmma: zero-pad to the next supported head dim (exact, beats XLA on V100).
    if cc < 75:
        return _pad_target(head_dim, cc) is not None
    return False


def ops_available(cc):
    """LayerNorm is Volta-capable; the CUTLASS gdp needs sm_75+."""
    syms = ("VoltaLayerNorm", "VoltaGdp") if cc >= 75 else ("VoltaLayerNorm",)
    ok = _load("layer_norm", cc, syms)
    if ok and cc < 75:
        ok = _load("gated_dual_proj", cc, ("VoltaGdpWmma",))
    return ok


@functools.partial(jax.jit, static_argnames=("sym", "sm_scale", "block_q", "block_k"))
def _attn_call(q, k, v, bias, kmask, *, sym, sm_scale, block_q, block_k):
    n, h, sq, d = q.shape
    # sequential: the template stack vmaps over templates, and the handlers take a fixed rank.
    return jax.ffi.ffi_call(sym, jax.ShapeDtypeStruct((n, h, sq, d), jnp.float16),
                            vmap_method="sequential")(
        q, k, v, bias, kmask,
        scale=np.float32(sm_scale),
        block_q=np.int64(block_q), block_k=np.int64(block_k))


def volta_attention(q, k, v, mask_bias, nonbatched_bias, scale, cc,
                    block_q=64, block_k=32):
    """Drop-in for tri_flash.pallas_attention. q/k/v [b,h,S,c] float16."""
    in_dtype = q.dtype
    b, h, sq, c = q.shape
    sk = k.shape[2]
    assert c == v.shape[-1], "kernel needs key_dim == value_dim"
    pad_to = None
    if c not in _native_dims(cc):
        pad_to = _pad_target(c, cc)
        assert pad_to is not None, f"unsupported head_dim {c}"
        pad = [(0, 0)] * 3 + [(0, pad_to - c)]
        q, k, v = (jnp.pad(t.astype(jnp.float16), pad) for t in (q, k, v))
    if cc >= 75 and (pad_to or c) == 64 and (block_q, block_k) == (64, 32):
        block_k = 64    # the mma kernel has no (64, 64, 32) instantiation
    # Template pointwise attention shares one mask row across the batch; the kernel
    # indexes the mask per row, so broadcast instead of reading past its end.
    kmask = jnp.broadcast_to(mask_bias[:, 0, 0, :] > -1e3, (b, sk)).astype(jnp.uint8)
    bias = (jnp.zeros((h, sq, sk), jnp.float16) if nonbatched_bias is None
            else nonbatched_bias.astype(jnp.float16))
    out = _attn_call(q.astype(jnp.float16), k.astype(jnp.float16),
                     v.astype(jnp.float16), bias, kmask,
                     sym=_attn_symbol(cc), sm_scale=float(scale),
                     block_q=block_q, block_k=block_k)
    if pad_to is not None:
        out = out[..., :c]
    return out.astype(in_dtype)


@functools.partial(jax.jit, static_argnames=("eps",))
def _ln_call(x, scale, offset, *, eps):
    m, c = x.shape
    return jax.ffi.ffi_call(
        "VoltaLayerNorm", jax.ShapeDtypeStruct((m, c), jnp.float16),
        vmap_method="sequential")(
            x, scale, offset, eps=np.float32(eps))


def volta_layer_norm(x, scale, offset, *, eps=1e-5):
    """Drop-in for tri_mul.pallas_layer_norm. Normalises the last axis."""
    c = x.shape[-1]
    out = _ln_call(x.reshape(-1, c).astype(jnp.float16),
                   scale.astype(jnp.float32), offset.astype(jnp.float32), eps=eps)
    return out.reshape(x.shape).astype(x.dtype)


@functools.partial(jax.jit, static_argnames=("sym",))
def _gdp_call(x, wp, bp, wg, bg, mask, *, sym):
    m = x.shape[0]
    n = wp.shape[1]
    return jax.ffi.ffi_call(sym, jax.ShapeDtypeStruct((m, n), jnp.float16),
                            vmap_method="sequential")(
        x, wp, bp, wg, bg, mask)


def volta_gated_dual_proj(x, wp, bp, wg, bg, mask, cc, *, split=True,
                          channel_major=False):
    """Drop-in for tri_mul.gated_dual_proj. x [M,K]; wp/wg [K,P]; mask [M]."""
    sym = "VoltaGdp" if cc >= 75 else "VoltaGdpWmma"
    f16 = lambda t: t.astype(jnp.float16)
    if not split:
        return _gdp_call(f16(x), f16(wp), f16(bp), f16(wg), f16(bg),
                         f16(mask), sym=sym).astype(x.dtype)
    ci = wp.shape[1] // 2
    halves = []
    for lo, hi in ((0, ci), (ci, 2 * ci)):
        halves.append(_gdp_call(f16(x), f16(wp[:, lo:hi]), f16(bp[lo:hi]),
                                f16(wg[:, lo:hi]), f16(bg[lo:hi]), f16(mask),
                                sym=sym).astype(x.dtype))
    if channel_major:
        # XLA would transpose row-major halves for the einsum anyway.
        halves = [h.T for h in halves]
    return halves[0], halves[1]
