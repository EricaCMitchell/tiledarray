#!/usr/bin/env python3
# This file is a part of TiledArray.
# Copyright (C) 2026  Virginia Tech
#
# ad_verify_common.py
# Utilities that verify_jax.py and verify_torch.py share, with no dependency on
# either framework: golden.json decoding, the Check comparison class, the
# tolerances, and the scenario driver. Each verifier imports these and gives its
# own REGISTRY of scenario functions.

import json
import sys

import numpy as np

# --------------------------------------------------------------------------- #
# Tolerances.
# --------------------------------------------------------------------------- #
TIGHT = dict(rtol=1e-11, atol=1e-12)
LOOSE = dict(rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- #
# golden.json decoding — returns NumPy types (framework-neutral).
# --------------------------------------------------------------------------- #
def load(path):
    with open(path) as fh:
        return json.load(fh)


def tensor_np(obj):
    """Decode {shape, complex, data} → NumPy array (row-major, complex128 if complex)."""
    data = np.asarray(obj["data"], dtype=np.float64)
    if obj["complex"]:
        data = data[0::2] + 1j * data[1::2]
    return data.reshape(obj["shape"])


def scalar(obj):
    """Decode a golden scalar: real float → float; {re, im} dict → complex."""
    if isinstance(obj, dict):
        return complex(obj["re"], obj["im"])
    return float(obj)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
class Check:
    """One labelled (computed vs golden) comparison within a scenario."""

    def __init__(self, label, computed, golden, tol=TIGHT):
        c = np.asarray(computed).ravel()
        g = np.asarray(golden).ravel()
        self.label = label
        self.tol = tol
        if c.shape != g.shape:
            self.ok = False
            self.max_abs = self.max_rel = float("inf")
            self.note = f"shape {c.shape} vs {g.shape}"
            return
        diff = np.abs(c - g)
        self.max_abs = float(diff.max()) if diff.size else 0.0
        denom = np.abs(g)
        self.max_rel = float((diff / np.maximum(denom, 1e-300)).max()) if g.size else 0.0
        self.ok = bool(np.allclose(c, g, **tol))
        self.note = ""

    @classmethod
    def assertion(cls, label, ok, max_abs, note=""):
        """Pass or fail with a precomputed magnitude, for example a negative
        control. Does not call allclose."""
        self = cls.__new__(cls)
        self.label = label
        self.tol = {}
        self.ok = bool(ok)
        self.max_abs = float(max_abs)
        self.max_rel = float("nan")
        self.note = note
        return self


# --------------------------------------------------------------------------- #
# Per-file scenario registry
# --------------------------------------------------------------------------- #
def make_scenario_registry():
    """Return (registry dict, @scenario decorator) for a per-file scenario set."""
    registry = {}

    def scenario(name):
        def deco(fn):
            registry[name] = fn
            return fn
        return deco

    return registry, scenario


# --------------------------------------------------------------------------- #
# Table-printing driver
# --------------------------------------------------------------------------- #
def run_scenarios(doc, registry, backend=""):
    """Iterate golden scenarios, call registered handlers, print check table.
    Returns True iff every check passes (missing scenario counts as failure)."""
    scenarios = doc["scenarios"]
    label = f" [{backend}]" if backend else ""
    hdr = f"{'scenario':28s} {'check':22s} {'max|abs|':>11s} {'max|rel|':>11s}  result"
    print(f"{hdr}{label}")
    print("-" * len(hdr))

    all_ok = True
    missing = []
    for s in scenarios:
        name = s["name"]
        fn = registry.get(name)
        if fn is None:
            missing.append(name)
            print(f"{name:28s} {'(no oracle)':22s} {'':>11s} {'':>11s}  SKIP")
            all_ok = False
            continue
        try:
            checks = fn(s)
        except Exception as exc:
            print(f"{name:28s} {'(exception)':22s} {'':>11s} {'':>11s}  FAIL  {exc}")
            all_ok = False
            continue
        for c in checks:
            status = "PASS" if c.ok else "FAIL"
            note = f"  {c.note}" if c.note else ""
            print(f"{name:28s} {c.label:22s} {c.max_abs:11.2e} {c.max_rel:11.2e}  "
                  f"{status}{note}")
            all_ok = all_ok and c.ok

    print("-" * len(hdr))
    if missing:
        print(f"WARNING: {len(missing)} scenario(s) had no {backend or 'oracle'}: {missing}")
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return all_ok


def main_driver(registry, backend=""):
    """Standard entry point: read golden path from argv[1], run scenarios."""
    path = sys.argv[1] if len(sys.argv) > 1 else "golden.json"
    doc = load(path)
    return 0 if run_scenarios(doc, registry, backend) else 1
