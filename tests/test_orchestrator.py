import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.documentation_agent import DiagnosticOrchestrator

print("=" * 55)
print("Phase 5 — Full Pipeline Test (TC001)")
print("=" * 55)

orch = DiagnosticOrchestrator()

report = orch.run(
    raw_input="4-month history of fatigue, weight gain, feeling cold all the time, constipation, low mood. Hair thinning and dry skin.",
    case_id="TC001",
    age=44,
    sex="F",
    pmh=[],
    medications=[]
)

print("\n" + "=" * 55)
print("DIAGNOSTIC REPORT")
print("=" * 55)
print(f"Case ID     : {report.case_id}")
print(f"Generated   : {report.generated_at}")

print(f"\nDifferentials ({len(report.differentials)}):")
for i, d in enumerate(report.differentials, 1):
    print(f"  {i}. {d.diagnosis[:80]}")
    print(f"     Confidence: {d.confidence*100:.0f}%")

print(f"\nUrgent alerts ({len(report.urgent_alerts)}):")
if report.urgent_alerts:
    for a in report.urgent_alerts:
        print(f"  ! {a}")
else:
    print("  None")

print(f"\nInvestigations ({len(report.test_recommendations)}):")
for i, t in enumerate(report.test_recommendations, 1):
    print(f"  {i}. {t.test_name[:80]}")

print(f"\nAudit steps  : {[a.step for a in report.audit_trail]}")
print(f"\nDisclaimer   : {report.disclaimer}")
print("\nPhase 5 complete.")

print("\n" + "=" * 55)
print("TC002 — Type 2 Diabetes")
print("=" * 55)

from agents.documentation_agent import DiagnosticOrchestrator
orch2 = DiagnosticOrchestrator()

report2 = orch2.run(
    raw_input="6-week history of excessive thirst, frequent urination, blurred vision, weight loss, tingling in both feet.",
    case_id="TC002",
    age=61,
    sex="M",
    pmh=["Hypertension", "Hypercholesterolaemia"],
    medications=["Amlodipine 5mg", "Atorvastatin 20mg"],
    vitals={"rbg": 18.4}
)

print(f"Differentials : {[d.diagnosis for d in report2.differentials]}")
print(f"Investigations: {[t.test_name for t in report2.test_recommendations]}")
print(f"Urgent alerts : {report2.urgent_alerts}")
