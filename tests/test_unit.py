import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.models import (
    Symptom, PatientCase, Differential,
    TestRecommendation, AuditEntry, DiagnosticReport
)
from agents.preprocessing_agent import PreprocessingAgent
from tools.safety_gate import SafetyGate


# ─────────────────────────────────────────
#  DATA MODEL TESTS
# ─────────────────────────────────────────

def test_symptom_creation():
    s = Symptom(raw_text="fatigue", hpo_code="HP:0012378", hpo_term="Fatigue")
    assert s.raw_text == "fatigue"
    assert s.hpo_code == "HP:0012378"
    assert s.is_negated == False

def test_symptom_negation_default_false():
    s = Symptom(raw_text="chest pain")
    assert s.is_negated == False

def test_patient_case_creation():
    case = PatientCase(case_id="TC001", age=44, sex="F", raw_input="fatigue")
    assert case.case_id == "TC001"
    assert case.age == 44
    assert case.sex == "F"
    assert case.symptoms == []
    assert case.medications == []
    assert case.pmh == []

def test_diagnostic_report_has_disclaimer():
    report = DiagnosticReport(case_id="TEST")
    assert len(report.disclaimer) > 0
    assert "clinician" in report.disclaimer.lower()

def test_diagnostic_report_generated_at():
    report = DiagnosticReport(case_id="TEST")
    assert report.generated_at is not None
    assert len(report.generated_at) > 0

def test_audit_entry_defaults():
    entry = AuditEntry(step="preprocessing")
    assert entry.success == True
    assert entry.error is None
    assert entry.step == "preprocessing"

def test_differential_confidence_range():
    d = Differential(diagnosis="Hypothyroidism", confidence=0.87)
    assert 0.0 <= d.confidence <= 1.0

def test_test_recommendation_priority():
    t = TestRecommendation(test_name="TSH", priority="urgent")
    assert t.priority == "urgent"
    assert t.test_name == "TSH"


# ─────────────────────────────────────────
#  PREPROCESSING AGENT TESTS
# ─────────────────────────────────────────

def test_preprocessing_extracts_fatigue():
    agent = PreprocessingAgent()
    case, audit = agent.run(
        raw_input="fatigue and weight gain",
        case_id="TEST", age=44, sex="F"
    )
    terms = [s.hpo_term for s in case.symptoms]
    assert "Fatigue" in terms

def test_preprocessing_extracts_multiple_symptoms():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="fatigue, weight gain, cold intolerance, constipation",
        case_id="TEST", age=44, sex="F"
    )
    assert len(case.symptoms) >= 3

def test_preprocessing_negation_chest_pain():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="No chest pain. Fatigue present.",
        case_id="TEST", age=44, sex="F"
    )
    negated = [s.hpo_term for s in case.symptoms if s.is_negated]
    assert "Chest pain" in negated

def test_preprocessing_negation_does_not_affect_other_symptoms():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="No chest pain. Fatigue present.",
        case_id="TEST", age=44, sex="F"
    )
    active = [s.hpo_term for s in case.symptoms if not s.is_negated]
    assert "Fatigue" in active

def test_preprocessing_audit_entry():
    agent = PreprocessingAgent()
    _, audit = agent.run(
        raw_input="fatigue, weight gain",
        case_id="TEST", age=44, sex="F"
    )
    assert audit.step == "preprocessing"
    assert audit.success == True

def test_preprocessing_empty_input():
    agent = PreprocessingAgent()
    case, audit = agent.run(
        raw_input="patient attended for review",
        case_id="TEST", age=50, sex="M"
    )
    assert case.case_id == "TEST"
    assert audit.success == True

def test_preprocessing_tc001_hypothyroidism():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="fatigue, weight gain, cold intolerance, constipation, dry skin",
        case_id="TC001", age=44, sex="F"
    )
    terms = [s.hpo_term for s in case.symptoms]
    assert "Fatigue" in terms
    assert "Weight gain" in terms
    assert "Cold intolerance" in terms

def test_preprocessing_tc003_menorrhagia_not_negated():
    agent = PreprocessingAgent()
    case, _ = agent.run(
        raw_input="Fatigue, shortness of breath, palpitations. No chest pain. Heavy periods for 5 months. Denies fever.",
        case_id="TC003", age=34, sex="F"
    )
    menorrhagia = [s for s in case.symptoms if s.hpo_term == "Menorrhagia"]
    assert len(menorrhagia) > 0
    assert menorrhagia[0].is_negated == False


# ─────────────────────────────────────────
#  SAFETY GATE TESTS
# ─────────────────────────────────────────

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

def test_safety_gate_detects_pulmonary_embolism():
    gate = SafetyGate()
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Pulmonary Embolism", confidence=0.5)],
        urgent_alerts=[]
    )
    report = gate.validate(report)
    assert any("PULMONARY EMBOLISM" in a for a in report.urgent_alerts)

def test_safety_gate_no_false_red_flags():
    gate = SafetyGate()
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Hypothyroidism", confidence=0.9)],
        urgent_alerts=[]
    )
    report = gate.validate(report)
    assert len(report.urgent_alerts) == 0

def test_safety_gate_injects_tsh_for_hypothyroidism():
    gate = SafetyGate()
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Hypothyroidism", confidence=0.9)],
        test_recommendations=[]
    )
    report = gate.validate(report)
    test_names = [t.test_name for t in report.test_recommendations]
    assert "TSH" in test_names

def test_safety_gate_injects_hba1c_for_diabetes():
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
            Differential(diagnosis="Anaemia",        confidence=0.4),
        ],
        test_recommendations=[]
    )
    report = gate.validate(report)
    diagnoses = [d.diagnosis for d in report.differentials]
    assert "PCOS" not in diagnoses
    assert "Hypothyroidism" in diagnoses

def test_safety_gate_keeps_existing_investigations():
    gate = SafetyGate()
    existing = [TestRecommendation("TSH", "urgent", "existing")]
    report = DiagnosticReport(
        case_id="TEST",
        differentials=[Differential(diagnosis="Hypothyroidism", confidence=0.9)],
        test_recommendations=existing
    )
    report = gate.validate(report)
    assert len(report.test_recommendations) >= 1
    assert report.test_recommendations[0].test_name == "TSH"

def test_safety_gate_disclaimer_always_present():
    gate = SafetyGate()
    report = DiagnosticReport(case_id="TEST")
    report = gate.validate(report)
    assert "clinician" in report.disclaimer.lower()
    assert len(report.disclaimer) > 50

def test_safety_gate_disclaimer_content():
    gate = SafetyGate()
    report = DiagnosticReport(case_id="TEST")
    report = gate.validate(report)
    assert "AI" in report.disclaimer
    assert "clinician" in report.disclaimer.lower()


# ─────────────────────────────────────────
#  AUDIO TRANSCRIBER TESTS
# ─────────────────────────────────────────

def test_transcriber_extracts_age():
    from data_processing.audio_transcriber import AudioTranscriber
    t = AudioTranscriber.__new__(AudioTranscriber)
    result = t._extract_age("44 year old female with fatigue")
    assert result == 44

def test_transcriber_extracts_sex_female():
    from data_processing.audio_transcriber import AudioTranscriber
    t = AudioTranscriber.__new__(AudioTranscriber)
    result = t._extract_sex("she is a 44 year old woman")
    assert result == "F"

def test_transcriber_extracts_sex_male():
    from data_processing.audio_transcriber import AudioTranscriber
    t = AudioTranscriber.__new__(AudioTranscriber)
    result = t._extract_sex("he is a 61 year old man")
    assert result == "M"

def test_transcriber_extracts_bp():
    from data_processing.audio_transcriber import AudioTranscriber
    t = AudioTranscriber.__new__(AudioTranscriber)
    result = t._extract_vitals("blood pressure 148 over 92")
    assert "bp" in result
    assert result["bp"] == "148/92"

def test_transcriber_extracts_hr():
    from data_processing.audio_transcriber import AudioTranscriber
    t = AudioTranscriber.__new__(AudioTranscriber)
    result = t._extract_vitals("heart rate 88 beats per minute")
    assert "hr" in result
    assert result["hr"] == "88"

def test_transcriber_extracts_medications():
    from data_processing.audio_transcriber import AudioTranscriber
    t = AudioTranscriber.__new__(AudioTranscriber)
    result = t._extract_medications("patient is on amlodipine and metformin")
    assert "Amlodipine" in result
    assert "Metformin" in result

def test_transcriber_extracts_pmh():
    from data_processing.audio_transcriber import AudioTranscriber
    t = AudioTranscriber.__new__(AudioTranscriber)
    result = t._extract_pmh("known hypertension and diabetes")
    assert "Hypertension" in result
    assert "Diabetes" in result
