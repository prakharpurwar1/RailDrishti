"""End-to-end governance demo: inject drift, run the closed-loop supervisor cycle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raildrishti.governance.platform import GovernancePlatform
from raildrishti.governance.supervisor import AgenticSupervisor


def main():
    platform = GovernancePlatform()
    supervisor = AgenticSupervisor(platform)

    print("=" * 70)
    print("1. PLATFORM HEALTH (all governed models)")
    print("=" * 70)
    rep = platform.report()
    print(f"overall={rep['overall']}  healthy={rep['healthy']} degraded={rep['degraded']} "
          f"critical={rep['critical']}  audit_chain_intact={rep['audit_chain_intact']}")
    for name, h in rep["models"].items():
        print(f"  {name:24s} {h['sla']['status']:9s} {h['sla']['slas_met']}/{h['sla']['slas_total']} SLA  {h['metrics']}")

    print("\n" + "=" * 70)
    print("2. INJECT DRIFT into confirmation_predictor (two features shift)")
    print("=" * 70)
    platform.inject_drift("confirmation_predictor", "wl_position", magnitude=0.6)
    platform.inject_drift("confirmation_predictor", "booking_days_before", magnitude=0.5, seed=2)
    drift = platform.drift("confirmation_predictor")
    print(f"drift severity={drift['severity']}  n_drifted={drift['n_drifted']}  max_psi={drift['max_psi']}")
    for f, info in drift["features"].items():
        print(f"  {f:22s} PSI={info['psi']:.3f} {info['severity']:7s} "
              f"{info['ref_mean']} -> {info['cur_mean']}")

    print("\n" + "=" * 70)
    print("3. ROOT CAUSE ANALYSIS")
    print("=" * 70)
    rca = platform.root_cause("confirmation_predictor", performance_drop=0.08)
    print(f"origin={rca['origin']} confidence={rca['confidence']}")
    print(f"explanation: {rca['explanation']}")
    print(f"remediations: {rca['recommended_remediations']}")

    print("\n" + "=" * 70)
    print("4. AGENTIC SUPERVISOR CLOSED-LOOP CYCLE")
    print("=" * 70)
    result = supervisor.run_cycle()
    print(f"cycle={result['cycle']} alerts={result['n_alerts']} actions={len(result['actions'])} "
          f"resolved={result['resolved']}")
    for a in result["alerts"]:
        print(f"  ALERT [{a['severity']:8s}] {a['model']:24s} {a['type']}: {a['detail']}")
    for d in result["decisions"]:
        print(f"  DECIDE {d['model']:24s} -> {d['action']:10s} ({d['rationale']})")
    for act in result["actions"]:
        print(f"  ACT    {act['model']:24s} -> {act.get('action')} resolved={act.get('resolved')} "
              f"status={act.get('post_action_status')}")

    print("\n" + "=" * 70)
    print("5. POST-CYCLE HEALTH + AUDIT CHAIN")
    print("=" * 70)
    rep2 = platform.report()
    print(f"overall={rep2['overall']}  audit_chain_intact={rep2['audit_chain_intact']} "
          f"events={rep2['audit_events']}")
    print(f"confirmation_predictor live_version={rep2['models']['confirmation_predictor']['live_version']}")
    ok, n = platform.audit.verify_chain()
    print(f"audit chain verified={ok} records={n}")


if __name__ == "__main__":
    main()
