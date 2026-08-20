#!/usr/bin/env python3
# This file is a part of TiledArray.
# Copyright (C) 2026  Virginia Tech
#
# verify_jax.py
# JAX oracle for the AD dual-verification harness (tests/ad_verify/). It runs
# with verify_torch.py. The two oracles sit on opposite sides of the
# complex-conjugation seam.
#
# Reads golden.json from the C++ producer `ad_produce`, recomputes the value,
# JVP, VJP, and HVP of each scenario with jax.grad, jax.jvp, and jax.vjp, and
# compares them to the TiledArray tape. The tolerance depends on the scenario.
# JAX is an independent oracle, so agreement shows that the AD math is correct,
# and not only self-consistent. The exit status is nonzero on any mismatch.

import sys

import numpy as np
import jax

# float64 and complex128 match the C++ double and complex<double> precision.
# Without this line JAX uses float32 and every check fails at approximately
# 1e-7.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402  (must follow the x64 config)

from ad_verify_common import (  # noqa: E402
    tensor_np, scalar, Check, TIGHT, LOOSE,
    make_scenario_registry, main_driver,
)

# --------------------------------------------------------------------------- #
# JAX tensor decoder. Wraps the shared NumPy decoder with jnp.asarray.
# --------------------------------------------------------------------------- #
def tensor(obj):
    """Decode a {shape, complex, data} object into a jnp array (row-major)."""
    return jnp.asarray(tensor_np(obj))


# --------------------------------------------------------------------------- #
# JAX reference computations, one for each scenario name. Each one takes the
# decoded scenario dict and returns a list of Check objects. `inp`, `out`, and
# `par` hold the named tensors, scalars, and parameters.
# --------------------------------------------------------------------------- #
REGISTRY, scenario = make_scenario_registry()


def make_accessors(s):
    inp = {k: tensor(v) for k, v in s["inputs"].items()}
    return inp, s["outputs"], s["params"]


# ---- forward values ------------------------------------------------------- #
@scenario("subt")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", inp["A"] - inp["B"], tensor(out["C"]))]


@scenario("permute")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", jnp.transpose(inp["A"]), tensor(out["C"]))]


@scenario("mult")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", inp["A"] * inp["B"], tensor(out["C"]))]


@scenario("conj")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", jnp.conj(inp["A"]), tensor(out["C"]))]


@scenario("elementwise")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", inp["A"] ** 2, tensor(out["C"]))]


@scenario("trace")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", jnp.trace(inp["A"]), scalar(out["s"]))]


@scenario("sum")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", jnp.sum(inp["A"]), scalar(out["s"]))]


@scenario("squared_norm")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", jnp.vdot(inp["A"], inp["A"]).real, scalar(out["s"]))]


@scenario("norm2")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", jnp.linalg.norm(inp["A"]), scalar(out["s"]), LOOSE)]


@scenario("dot")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", jnp.sum(inp["A"] * inp["B"]), scalar(out["s"]))]


@scenario("inner_product")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", jnp.vdot(inp["A"], inp["B"]), scalar(out["s"]))]


# ---- forward mode / JVP --------------------------------------------------- #
def _jvp_checks(f, primals, tangents, out, want_tangent="tangent",
                primal_key="primal", tol=None):
    tol = TIGHT if tol is None else tol
    primal_out, tangent_out = jax.jvp(f, primals, tangents)
    g_primal = out[primal_key]
    g_tangent = out[want_tangent]
    gp = scalar(g_primal) if not (isinstance(g_primal, dict) and "data" in g_primal) else tensor(g_primal)
    gt = scalar(g_tangent) if not (isinstance(g_tangent, dict) and "data" in g_tangent) else tensor(g_tangent)
    return [Check(primal_key, primal_out, gp, tol),
            Check(want_tangent, tangent_out, gt, tol)]


@scenario("contract_jvp")
def _(s):
    inp, out, par = make_accessors(s)
    factor = par["factor"]
    f = lambda A, B: factor * jnp.einsum("ik,kj->ij", A, B)
    return _jvp_checks(f, (inp["A"], inp["B"]), (inp["dA"], inp["dB"]), out)


@scenario("add_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: A + B, (inp["A"], inp["B"]),
                       (inp["dA"], inp["dB"]), out)


@scenario("subt_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: A - B, (inp["A"], inp["B"]),
                       (inp["dA"], inp["dB"]), out)


@scenario("scale_jvp")
def _(s):
    inp, out, par = make_accessors(s)
    alpha = par["alpha"]
    return _jvp_checks(lambda A: alpha * A, (inp["A"],), (inp["dA"],), out)


@scenario("permute_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(jnp.transpose, (inp["A"],), (inp["dA"],), out)


@scenario("mult_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: A * B, (inp["A"], inp["B"]),
                       (inp["dA"], inp["dB"]), out)


@scenario("elementwise_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A: A ** 2, (inp["A"],), (inp["dA"],), out)


@scenario("trace_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(jnp.trace, (inp["A"],), (inp["dA"],), out)


@scenario("sum_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(jnp.sum, (inp["A"],), (inp["dA"],), out)


@scenario("squared_norm_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    f = lambda A: jnp.vdot(A, A).real
    return _jvp_checks(f, (inp["A"],), (inp["dA"],), out, tol=LOOSE)


@scenario("norm2_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(jnp.linalg.norm, (inp["A"],), (inp["dA"],), out, tol=LOOSE)


@scenario("dot_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: jnp.sum(A * B), (inp["A"], inp["B"]),
                       (inp["dA"], inp["dB"]), out)


# ---- complex forward mode / JVP ------------------------------------------- #
# The JVP does not depend on the gradient convention, so the plus/minus split
# does not appear and no adapter is necessary. jax.jvp is a direct oracle. These
# scenarios pin the feature that an op which is not holomorphic carries a
# conjugated tangent, conj(dA). jax.jvp gives the same result, because the JVP
# of jnp.conj conjugates the tangent.
@scenario("complex_conj_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(jnp.conj, (inp["A"],), (inp["dA"],), out)


@scenario("complex_dot_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: jnp.sum(A * B), (inp["A"], inp["B"]),
                       (inp["dA"], inp["dB"]), out)


@scenario("complex_inner_product_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: jnp.sum(jnp.conj(A) * B),
                       (inp["A"], inp["B"]), (inp["dA"], inp["dB"]), out)


# ---- reverse mode / VJP (real) -------------------------------------------- #
def _vjp_array(f, primals, cotangent, out, names, tol=None):
    tol = TIGHT if tol is None else tol
    _, vjp = jax.vjp(f, *primals)
    grads = vjp(cotangent)
    return [Check(n, g, tensor(out[n]), tol) for n, g in zip(names, grads)]


@scenario("contract_vjp")
def _(s):
    inp, out, par = make_accessors(s)
    factor = par["factor"]
    f = lambda A, B: factor * jnp.einsum("ik,kj->ij", A, B)
    return _vjp_array(f, (inp["A"], inp["B"]), tensor(s["inputs"]["Cbar"]),
                      out, ["Abar", "Bbar"])


@scenario("add_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _vjp_array(lambda A, B: A + B, (inp["A"], inp["B"]),
                      inp["Cbar"], out, ["Abar", "Bbar"])


@scenario("subt_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _vjp_array(lambda A, B: A - B, (inp["A"], inp["B"]),
                      inp["Cbar"], out, ["Abar", "Bbar"])


@scenario("mult_permute_scale_vjp")
def _(s):
    inp, out, par = make_accessors(s)
    alpha = par["alpha"]
    f = lambda A, B: alpha * jnp.transpose(A * B)
    return _vjp_array(f, (inp["A"], inp["B"]), inp["Cbar"], out, ["Abar", "Bbar"])


def _grad_check(f, A, out, key="grad", tol=None):
    tol = TIGHT if tol is None else tol
    return [Check(key, jax.grad(f)(A), tensor(out[key]), tol)]


@scenario("sum_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(jnp.sum, inp["A"], out)


@scenario("squared_norm_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(lambda A: jnp.vdot(A, A).real, inp["A"], out)


@scenario("norm2_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(jnp.linalg.norm, inp["A"], out, tol=LOOSE)


@scenario("trace_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(jnp.trace, inp["A"], out)


@scenario("elementwise_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(lambda A: jnp.sum(A ** 3), inp["A"], out)


@scenario("fan_out_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    P, Q = inp["P"], inp["Q"]
    return _grad_check(lambda A: jnp.sum(A * P + A * Q), inp["A"], out)


# ---- complex VJP seam ----------------------------------------------------- #
# Adapter: the tape uses the Re<X-bar, dX> pairing, and JAX uses a
# complex-linear vjp. To bridge them, seed jax.vjp with conj(s-bar) and
# conjugate each returned cotangent. For a real s-bar the conjugation is the
# identity, so it does not change the real scenarios. The negative control drops
# the output conjugation, and must then differ.
def _complex_vjp_checks(s, f):
    inp, out, par = make_accessors(s)
    sbar = scalar(par["sbar"])
    _, vjp = jax.vjp(f, inp["A"], inp["B"])
    ja, jb = vjp(np.conj(sbar))
    checks = [Check("Abar(+adapter)", jnp.conj(ja), tensor(out["Abar"])),
              Check("Bbar(+adapter)", jnp.conj(jb), tensor(out["Bbar"]))]
    # Negative control: without the adapter the seam must be visible.
    raw_gap = float(np.abs(np.asarray(ja).ravel()
                           - np.asarray(tensor(out["Abar"])).ravel()).max())
    checks.append(Check.assertion("seam(no-adapter differs)", raw_gap > 1e-3,
                                  raw_gap, "expect large gap"))
    return checks


@scenario("complex_dot_vjp")
def _(s):
    return _complex_vjp_checks(s, lambda A, B: jnp.sum(A * B))


@scenario("complex_inner_product_vjp")
def _(s):
    return _complex_vjp_checks(s, lambda A, B: jnp.sum(jnp.conj(A) * B))


# ---- the 1/2 z^2 convention litmus ---------------------------------------- #
# The litmus: the gradient of f(z) = 1/2 z^2 at z = 1+i is 1-i under the
# conjugating ("plus") convention that TA uses, and 1+i under the JAX
# convention.
@scenario("complex_half_sq_litmus")
def _(s):
    inp, out, _ = make_accessors(s)
    Z, Cbar = inp["Z"], inp["Cbar"]
    grad = tensor(out["grad"])
    f = lambda z: 0.5 * z ** 2
    _, vjp = jax.vjp(f, Z)
    (ja,) = vjp(jnp.conj(Cbar))  # seed jax.vjp with conj(s-bar)
    checks = [
        Check("grad(+adapter)", jnp.conj(ja), grad),
        Check("litmus(=1-i)", grad, jnp.full(grad.shape, 1.0 - 1.0j)),
    ]
    raw_gap = float(np.abs(np.asarray(ja).ravel()
                           - np.asarray(grad).ravel()).max())
    checks.append(Check.assertion("seam(no-adapter differs)", raw_gap > 1e-3,
                                  raw_gap, "minus conv. gives 1+i"))
    return checks


# ---- Hessian-vector product ----------------------------------------------- #
@scenario("hvp")
def _(s):
    inp, out, _ = make_accessors(s)
    X, v = inp["X"], inp["v"]
    E = lambda Z: jnp.sum(jnp.abs(Z @ Z) ** 2)  # ||X.X||^2, real
    g_jax, Hv_jax = jax.jvp(jax.grad(E), (X,), (v,))
    return [Check("g", g_jax, tensor(out["g"]), LOOSE),
            Check("Hv", Hv_jax, tensor(out["Hv"]), LOOSE)]


# ---- block-sparse contract VJP -------------------------------------------- #
@scenario("sparse_contract_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    f = lambda A, B: jnp.einsum("ik,kj->ij", A, B)
    return _vjp_array(f, (inp["A"], inp["B"]), inp["Cbar"], out, ["Abar", "Bbar"])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sys.exit(main_driver(REGISTRY, "jax"))
