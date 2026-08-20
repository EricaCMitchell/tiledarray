#!/usr/bin/env python3
# This file is a part of TiledArray.
# Copyright (C) 2026  Virginia Tech
#
# verify_torch.py
# PyTorch oracle for the AD dual-verification harness (tests/ad_verify/). It
# runs with verify_jax.py. The two oracles sit on opposite sides of the
# complex-conjugation seam. PyTorch uses the same conjugating ("plus")
# convention as the tape, so it is the positive control and matches directly.
# JAX uses the non-conjugating ("minus") convention and needs an adapter.
#
# Reads golden.json from the C++ producer `ad_produce`, recomputes the value,
# JVP, VJP, and HVP of each scenario with torch.func.{jvp,vjp,grad}, and
# compares them to the TiledArray tape. The tolerance depends on the scenario.
# The exit status is nonzero on any mismatch.
#
# Requires torch 2.0 or later, which has torch.func.

import sys

import numpy as np
import torch
import torch.func

# float64 and complex128 match the C++ double and complex<double> precision.
torch.set_default_dtype(torch.float64)

from ad_verify_common import (
    tensor_np, scalar, Check, TIGHT, LOOSE,
    make_scenario_registry, main_driver,
)

# --------------------------------------------------------------------------- #
# PyTorch tensor decoder and NumPy conversion.
# --------------------------------------------------------------------------- #
def tensor(obj):
    """Decode a {shape, complex, data} object into a torch tensor (row-major)."""
    arr = tensor_np(obj)
    if np.iscomplexobj(arr):
        return torch.from_numpy(arr.astype(np.complex128))
    return torch.from_numpy(arr.astype(np.float64))


def to_np(t):
    """Convert a torch tensor (including lazy-conj views) to a NumPy array."""
    if isinstance(t, torch.Tensor):
        return t.detach().resolve_conj().numpy()
    return np.asarray(t)


def make_accessors(s):
    inp = {k: tensor(v) for k, v in s["inputs"].items()}
    return inp, s["outputs"], s["params"]


# --------------------------------------------------------------------------- #
# PyTorch AD scenarios, one for each scenario name.
# --------------------------------------------------------------------------- #
REGISTRY, scenario = make_scenario_registry()


# ---- forward values ------------------------------------------------------- #
@scenario("subt")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", to_np(inp["A"] - inp["B"]), tensor_np(out["C"]))]


@scenario("permute")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", to_np(inp["A"].mT), tensor_np(out["C"]))]


@scenario("mult")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", to_np(inp["A"] * inp["B"]), tensor_np(out["C"]))]


@scenario("conj")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", to_np(torch.conj(inp["A"])), tensor_np(out["C"]))]


@scenario("elementwise")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("C", to_np(inp["A"] ** 2), tensor_np(out["C"]))]


@scenario("trace")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", to_np(torch.trace(inp["A"])), scalar(out["s"]))]


@scenario("sum")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", to_np(inp["A"].sum()), scalar(out["s"]))]


@scenario("squared_norm")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", to_np(torch.sum(inp["A"].conj() * inp["A"]).real),
                  scalar(out["s"]))]


@scenario("norm2")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", to_np(torch.linalg.norm(inp["A"])), scalar(out["s"]), LOOSE)]


@scenario("dot")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", to_np(torch.sum(inp["A"] * inp["B"])), scalar(out["s"]))]


@scenario("inner_product")
def _(s):
    inp, out, _ = make_accessors(s)
    return [Check("s", to_np(torch.sum(torch.conj(inp["A"]) * inp["B"])),
                  scalar(out["s"]))]


# ---- forward mode / JVP --------------------------------------------------- #
def _jvp_checks(f, primals, tangents, out, want_tangent="tangent",
                primal_key="primal", tol=TIGHT):
    primal_out, tangent_out = torch.func.jvp(f, primals, tangents)
    g_primal = out[primal_key]
    g_tangent = out[want_tangent]
    gp = tensor_np(g_primal) if isinstance(g_primal, dict) and "data" in g_primal else scalar(g_primal)
    gt = tensor_np(g_tangent) if isinstance(g_tangent, dict) and "data" in g_tangent else scalar(g_tangent)
    return [Check(primal_key, to_np(primal_out), gp, tol),
            Check(want_tangent, to_np(tangent_out), gt, tol)]


@scenario("contract_jvp")
def _(s):
    inp, out, par = make_accessors(s)
    factor = par["factor"]
    f = lambda A, B: factor * torch.einsum("ik,kj->ij", A, B)
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
    return _jvp_checks(lambda A: A.mT, (inp["A"],), (inp["dA"],), out)


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
    return _jvp_checks(torch.trace, (inp["A"],), (inp["dA"],), out)


@scenario("sum_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A: A.sum(), (inp["A"],), (inp["dA"],), out)


@scenario("squared_norm_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    f = lambda A: torch.sum(A.conj() * A).real
    return _jvp_checks(f, (inp["A"],), (inp["dA"],), out, tol=LOOSE)


@scenario("norm2_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(torch.linalg.norm, (inp["A"],), (inp["dA"],), out, tol=LOOSE)


@scenario("dot_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: torch.sum(A * B), (inp["A"], inp["B"]),
                       (inp["dA"], inp["dB"]), out)


# ---- complex forward mode / JVP ------------------------------------------- #
# The JVP does not depend on the gradient convention, so torch.func.jvp is a
# direct oracle, as jax.jvp is. These scenarios pin the feature that an op which
# is not holomorphic carries a conjugated tangent, conj(dA). torch.func.jvp
# gives the same result, because the forward-mode rule for torch.conj
# conjugates the tangent.
@scenario("complex_conj_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(torch.conj, (inp["A"],), (inp["dA"],), out)


@scenario("complex_dot_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: torch.sum(A * B), (inp["A"], inp["B"]),
                       (inp["dA"], inp["dB"]), out)


@scenario("complex_inner_product_jvp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _jvp_checks(lambda A, B: torch.sum(torch.conj(A) * B),
                       (inp["A"], inp["B"]), (inp["dA"], inp["dB"]), out)


# ---- reverse mode / VJP (real) -------------------------------------------- #
def _vjp_array(f, primals, cotangent, out, names, tol=TIGHT):
    _, vjp_fn = torch.func.vjp(f, *primals)
    grads = vjp_fn(cotangent)
    return [Check(n, to_np(g), tensor_np(out[n]), tol) for n, g in zip(names, grads)]


def _grad_check(f, A, out, key="grad", tol=TIGHT):
    g = torch.func.grad(f)(A)
    return [Check(key, to_np(g), tensor_np(out[key]), tol)]


@scenario("contract_vjp")
def _(s):
    inp, out, par = make_accessors(s)
    factor = par["factor"]
    f = lambda A, B: factor * torch.einsum("ik,kj->ij", A, B)
    return _vjp_array(f, (inp["A"], inp["B"]), inp["Cbar"], out, ["Abar", "Bbar"])


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
    f = lambda A, B: alpha * (A * B).mT
    return _vjp_array(f, (inp["A"], inp["B"]), inp["Cbar"], out, ["Abar", "Bbar"])


@scenario("sum_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(lambda A: A.sum(), inp["A"], out)


@scenario("squared_norm_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(lambda A: torch.sum(A.conj() * A).real, inp["A"], out)


@scenario("norm2_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(torch.linalg.norm, inp["A"], out, tol=LOOSE)


@scenario("trace_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(torch.trace, inp["A"], out)


@scenario("elementwise_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    return _grad_check(lambda A: torch.sum(A ** 3), inp["A"], out)


@scenario("fan_out_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    P, Q = inp["P"], inp["Q"]
    return _grad_check(lambda A: torch.sum(A * P + A * Q), inp["A"], out)


# ---- complex VJP seam ----------------------------------------------------- #
# PyTorch uses the conjugating ("plus") convention, the same as the TiledArray
# tape, so the comparison is direct and needs no adapter. This is the positive
# control.
#
# Negative control: rebuild the "minus" (JAX) value from the torch vjp with
# minus(s̄) = conj( torch.vjp(f)(conj(s̄)) ), and assert that it differs from the
# tape. This shows that PyTorch is on the plus side, and that the direct match
# above means something.
def _complex_vjp_checks(s, f):
    inp, out, par = make_accessors(s)
    sbar = scalar(par["sbar"])
    sbar_t = torch.tensor(sbar, dtype=torch.complex128)

    _, vjp_fn = torch.func.vjp(f, inp["A"], inp["B"])
    ja, jb = vjp_fn(sbar_t)

    checks = [Check("Abar(direct)", to_np(ja), tensor_np(out["Abar"])),
              Check("Bbar(direct)", to_np(jb), tensor_np(out["Bbar"]))]

    # Rebuild the minus value: seed with conj(s̄), then conjugate the result.
    ma, _ = vjp_fn(torch.conj(sbar_t))
    minus_ref = torch.conj(ma)
    raw_gap = float(np.abs(to_np(minus_ref).ravel()
                           - tensor_np(out["Abar"]).ravel()).max())
    checks.append(Check.assertion("seam(minus differs)", raw_gap > 1e-3, raw_gap))
    return checks


@scenario("complex_dot_vjp")
def _(s):
    return _complex_vjp_checks(s, lambda A, B: torch.sum(A * B))


@scenario("complex_inner_product_vjp")
def _(s):
    return _complex_vjp_checks(s, lambda A, B: torch.sum(torch.conj(A) * B))


# ---- 1/2 z^2 convention litmus -------------------------------------------- #
# Under the plus convention of PyTorch, vjp_fn(Cbar) gives the same 1-i as the
# tape, with no adapter. The negative control rebuilds the minus value (1+i) and
# asserts that it differs.
@scenario("complex_half_sq_litmus")
def _(s):
    inp, out, _ = make_accessors(s)
    Z, Cbar_t = inp["Z"], inp["Cbar"]
    grad_golden = tensor_np(out["grad"])

    f = lambda z: 0.5 * z ** 2
    _, vjp_fn = torch.func.vjp(f, Z)
    (ja,) = vjp_fn(Cbar_t)

    checks = [
        Check("litmus(=1-i)", to_np(ja), grad_golden),
        Check("litmus const(=1-i)", to_np(ja),
              np.full(grad_golden.shape, 1.0 - 1.0j)),
    ]

    # Rebuild the minus value: conj( vjp_fn(conj(C̄)) ).
    (ma,) = vjp_fn(torch.conj(Cbar_t))
    minus_ref = torch.conj(ma)
    raw_gap = float(np.abs(to_np(minus_ref).ravel() - grad_golden.ravel()).max())
    checks.append(Check.assertion("seam(minus differs)", raw_gap > 1e-3,
                                  raw_gap, "minus conv. gives 1+i"))
    return checks


# ---- Hessian-vector product ----------------------------------------------- #
@scenario("hvp")
def _(s):
    inp, out, _ = make_accessors(s)
    X, v = inp["X"], inp["v"]
    E = lambda Z: torch.sum(torch.abs(Z @ Z) ** 2)
    g_torch, Hv_torch = torch.func.jvp(torch.func.grad(E), (X,), (v,))
    return [Check("g", to_np(g_torch), tensor_np(out["g"]), LOOSE),
            Check("Hv", to_np(Hv_torch), tensor_np(out["Hv"]), LOOSE)]


# ---- block-sparse contract VJP -------------------------------------------- #
@scenario("sparse_contract_vjp")
def _(s):
    inp, out, _ = make_accessors(s)
    f = lambda A, B: torch.einsum("ik,kj->ij", A, B)
    return _vjp_array(f, (inp["A"], inp["B"]), inp["Cbar"], out, ["Abar", "Bbar"])


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sys.exit(main_driver(REGISTRY, "torch"))
