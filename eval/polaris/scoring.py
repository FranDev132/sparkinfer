#!/usr/bin/env python3
"""Self-contained scoring script that runs inside a Polaris TDX enclave.

This script replicates the logic from evaluate_dual.sh lines 148-212 in pure
Python (stdlib only — no dependencies). It reads the eval result JSON from
/submission/result.json (mounted by the Polaris attest workflow), applies the
correctness gate and no-regression guard, and outputs the final verdict as JSON
to stdout.

The DCAP quote proves this exact script, with these exact inputs, produced
this exact output — hardware-attested scoring.
"""

import json
import re
import sys

TOP1_BAR = 0.90
KL_BAR = 0.20

GUARD_CTX_KEYS = [
    "guard_128_pass",
    "guard_512_pass",
    "guard_4k_pass",
    "guard_16k_pass",
    "guard_32k_pass",
]

CTX_LABEL_MAP = {
    "guard_128_pass": "128",
    "guard_512_pass": "512",
    "guard_4k_pass": "4k",
    "guard_16k_pass": "16k",
    "guard_32k_pass": "32k",
}


def load_result(path: str = "/submission/result.json") -> dict:
    """Load the eval result JSON from the submission mount."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({
            "label": "REJECT",
            "pass": False,
            "reason": f"scoring: failed to load result.json: {e}",
        }))
        sys.exit(1)


def score(result: dict) -> dict:
    """Apply correctness gate + no-regression guard and return the final verdict.

    Args:
        result: The merged RESULT_JSON dict (primary + guard fields).

    Returns:
        A verdict dict with at least: label, pass, reason.
    """
    primary = result.get("primary", result)
    guard = result.get("guard") or {}

    # An empty primary is an infra failure — fail closed.
    if not primary:
        return {
            "label": "REJECT",
            "pass": False,
            "reason": "primary (Qwen3.6) eval produced no verdict (infra error)",
        }

    # ---- Correctness gate ----
    top1 = float(primary.get("top1", 0))
    kl = float(primary.get("kl", 99))
    acc_ok = top1 >= TOP1_BAR and kl <= KL_BAR

    if not acc_ok:
        reasons = []
        if top1 < TOP1_BAR:
            reasons.append(f"top1={top1:.4f} (need >= {TOP1_BAR})")
        if kl > KL_BAR:
            reasons.append(f"kl={kl:.4f} (need <= {KL_BAR})")
        return {
            "label": "REJECT",
            "pass": False,
            "reason": "correctness gate failed: " + "; ".join(reasons),
        }

    # ---- No-regression guard ----
    # Every measured context must hold >= tolerance vs main branch baseline.
    present = [k for k in GUARD_CTX_KEYS if k in guard]
    # bool(present) matters: all([]) is True, so a guard record that measured NO context would
    # otherwise read as a pass. bool(guard) below catches a wholly EMPTY guard dict but not a
    # PARTIAL one carrying only top1/kl and no context keys -- which is exactly the shape a
    # crashed or truncated guard run produces. Fail closed on incomplete evidence, the same way
    # this function already fails closed on missing and on bad evidence.
    speed_ok = bool(present) and all(guard.get(k, True) for k in present)
    g_top1 = float(guard.get("top1", 0))
    g_kl = float(guard.get("kl", 99))
    g_acc_ok = g_top1 >= TOP1_BAR and g_kl <= KL_BAR
    guard_ok = bool(guard) and speed_ok and g_acc_ok

    regressed = [CTX_LABEL_MAP[k] for k in present if not guard.get(k, True)]

    # Build the final result — carry primary metrics verbatim, overlay guard.
    final = dict(primary)
    final["guard"] = {
        k: guard[k]
        for k in (
            "pass", "top1", "kl", "label",
            "ctx_128_tps", "ctx_512_tps", "ctx_4096_tps",
            "ctx_16384_tps", "ctx_32768_tps",
            "guard_128_pass", "guard_512_pass", "guard_4k_pass",
            "guard_16k_pass", "guard_32k_pass",
        )
        if k in guard
    }
    final["guard"]["speed_ok"] = speed_ok
    final["guard"]["accuracy_ok"] = g_acc_ok

    if not guard_ok:
        reasons = []
        if regressed:
            reasons.append(
                "Qwen3-30B decode regressed at context(s): " + ", ".join(regressed)
            )
        if not g_acc_ok:
            reasons.append(
                f"Qwen3-30B accuracy broke (top1={g_top1} need>={TOP1_BAR}, "
                f"kl={g_kl} need<={KL_BAR})"
            )
        if not bool(guard):
            reasons.append("Qwen3-30B guard produced no verdict (infra error)")
        if not reasons:
            reasons.append("Qwen3-30B no-regression guard failed")
        final["label"] = "REJECT"
        final["pass"] = False
        final["reason"] = "no-regression guard: " + "; ".join(reasons)
        final["guard_regression_labels"] = [
            "regression-qwen3-" + c for c in regressed
        ]

    return final


def score_dflash(result: dict) -> dict:
    """Score a DFlash bot result (measurements.dflash / measurements.qwen_guard shape, from
    pr_dflash_bot.py's eval_dflash_on_box()) — distinct schema from the AR bot's primary/guard
    RESULT_JSON, so score() above (which reads primary.top1/kl) can't handle it: those fields
    don't exist in this shape, silently default to 0/99, and score() falls through to an
    identical REJECT regardless of the real input — which is exactly what produced a constant
    result_sha256 across genuinely different DFlash evals before this function existed.

    Gate: DFlash accuracy (SPEC_AGREE must be exact) + the Qwen3.5/3.6 no-regression guard.
    """
    dflash = result.get("dflash") or {}
    guard = result.get("qwen_guard") or {}

    if not dflash:
        return {
            "label": "REJECT",
            "pass": False,
            "reason": "dflash measurements missing (infra error)",
        }

    spec_agree = str(dflash.get("spec_agree", ""))
    m = re.search(r"=\s*([0-9.]+)\s*$", spec_agree.strip())
    ratio = float(m.group(1)) if m else 0.0
    acc_ok = ratio >= 0.999
    guard_ok = bool(guard) and bool(guard.get("ok", False))

    final = dict(dflash)
    final["guard"] = guard

    if not acc_ok:
        final["label"] = "REJECT"
        final["pass"] = False
        final["reason"] = f"DFlash accuracy gate failed: SPEC_AGREE={ratio:.4f} (need >= 0.999)"
        return final
    if not guard_ok:
        reasons = guard.get("problems") or ["qwen3.5/3.6 guard produced no verdict (infra error)"]
        final["label"] = "REJECT"
        final["pass"] = False
        final["reason"] = "qwen3.5/qwen3.6 no-regression guard failed: " + "; ".join(reasons[:6])
        return final

    final["label"] = "ok"
    final["pass"] = True
    final["reason"] = f"DFlash accuracy (SPEC_AGREE={ratio:.4f}) + qwen3.5/3.6 guard both pass"
    return final


def main():
    result = load_result()
    verdict = score_dflash(result) if "dflash" in result else score(result)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
