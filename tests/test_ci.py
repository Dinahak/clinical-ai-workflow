import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.models import (
    Symptom, PatientCase, Differential,
    TestRecommendation, AuditEntry, DiagnosticReport
)
from agents.preprocessing_agent import PreprocessingAgent
from tools.safety_gate import SafetyGate


def test_symptom_creation():
    s = Symptom(raw_text="fatigue", hpo_code="HP:0012378", hpo_term="Fatigue")
    assert s.raw_text == "fatigue"
    assert s.is_negated == False

def test_patient_case_creation():
    case = PatientCase(case_id="TC001", age=44, sex="F", raw_input="fatigue")
    assert case.case_id == "TC001"
    assert case.age == 44

def test_diagnostic_report_disclaimer():
    report = DiagnosticReport(case_id="TEST")
    assert "clinician" in report.disclaimer.lower()

def test_audit_entry_defaults():
    entry = AuditEntry(step="preprocessing")
    assert entry.success == True
    assert entry.step == "preprocessing"

def test_differential_confidence():
    d = Differential(diagnosis="Hypothyroidism", confidence=0.87)
    assert 0.0 <= d.confidence <= 1.0

def test_preprocessing_extracts_fatigue():
    agent = PreprocessingAgent()
    case, audit = agent.run(
        raw_input="fatigue and weight gain",
        case_id="TEST", age=44, sex="F"
    )
    terms = [s.hpo_term for s in case.symptoms]
    assert "Fatigue" in terms
    assert audit.success == True

def test_preprocessing_multiple_symptoms():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="fatigue, weight gain, cold intolerance, constipation",
        case_id="TEST", age=44, sex="F"
    )
    assert len(case.symptoms) >= 3

def test_preprocessing_negation():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="No chest pain. Fatigue present.",
        case_id="TEST", age=44, sex="F"
    )
    negated = [s.hpo_term for s in case.symptoms if s.is_negated]
    assert "Chest pain" in negated

def test_preprocessing_tc001():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="fatigue, weight gain, cold intolerance, constipation, dry skin",
        case_id="TC001", age=44, sex="F"
    )
    terms = [s.hpo_term for s in case.symptoms]
    assert "Fatigue" in terms
    assert "Weight gain" in terms
    assert "Cold intolerance" in terms

def test_preprocessing_menorrhagia_not_negated():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="Fatigue, shortness of breath. No chest pain. Heavy periods for 5 months. Denies fever.",
        case_id="TC003", age=34, sex="F"
    )
    menorrhagia = [s for s in case.symptoms if s.hpo_term == "Menorrhagia"]
    assert len(menorrhagia) > 0
    assert menorrhagia[0].is_negated == False

def test_safety_gate_detects_sepsis():
    gate = SafetyGate()
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Sepsis", confidence=0.7)],
        urgent_alerts=[]
    )
    report = gate.validate(report)
    assert len(report.urgent_alerts) > 0
    assert "SEPSIS" in report.urgent_alerts[0]

def test_safety_gate_no_false_red_flags():
    gate = SafetyGate()
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Hypothyroidism", confidence=0.9)],
        urgent_alerts=[]
    )
    report = gate.validate(report)
    assert len(report.urgent_alerts) == 0

def test_safety_gate_injects_tsh():
    gate = SafetyGate()
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Hypothyroidism", confidence=0.9)],
        test_recommendations=[]
    )
    report = gate.validate(report)
    test_names = [t.test_name for t in report.test_recommendations]
    assert "TSH" in test_names

def test_safety_gate_injects_hba1c():
    gate = SafetyGate()
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Type 2 Diabetes", confidence=0.9)],
        test_recommendations=[]
    )
    report = gate.validate(report)
    test_names = [t.test_name for t in report.test_recommendations]
    assert "HbA1c" in test_names

def test_safety_gate_removes_low_confidence():
    gate = SafetyGate(confidence_threshold=0.1)
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[
            Differential(diagnosis="Hypothyroidism", confidence=0.9),
            Differential(diagnosis="PCOS",           confidence=0.05),
        ],
        test_recommendations=[]
    )
    report = gate.validate(report)
    diagnoses = [d.diagnosis for d in report.differentials]
    assert "PCOS" not in diagnoses
    assert "Hypothyroidism" in diagnoses

def test_safety_gate_disclaimer():
    gate = SafetyGate()
    report = DiagnosticReport(case_id="TEST")
    report = gate.validate(report)
    assert "clinician" in report.disclaimer.lower()
    assert "AI" in report.disclaimer
