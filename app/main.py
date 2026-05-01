# app/main.py

import streamlit as st
import sys, os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.documentation_agent import DiagnosticOrchestrator
from tools.safety_gate import SafetyGate

st.set_page_config(
    page_title="Clinical AI — Family Medicine",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Clinical AI — Family Medicine Assistant")
st.caption(
    "AI-powered clinical decision support. "
    "All outputs require review by a qualified clinician."
)
st.divider()

# ─────────────────────────────────────────
#  SIDEBAR — PATIENT DETAILS
# ─────────────────────────────────────────
with st.sidebar:
    st.header("Patient Details")
    case_id = st.text_input("Case ID", value="CASE001")
    age     = st.number_input("Age", min_value=1, max_value=120, value=44)
    sex     = st.selectbox("Sex", ["F", "M", "Other"])

    st.subheader("Past Medical History")
    pmh_input = st.text_area(
        "One condition per line",
        placeholder="Hypertension\nHypercholesterolaemia",
        height=100
    )

    st.subheader("Current Medications")
    med_input = st.text_area(
        "One medication per line",
        placeholder="Amlodipine 5mg OD\nAtorvastatin 20mg ON",
        height=100
    )

    st.subheader("Vitals (optional)")
    col1, col2 = st.columns(2)
    with col1:
        hr   = st.text_input("HR (bpm)", placeholder="88")
        bp   = st.text_input("BP",       placeholder="148/92")
    with col2:
        temp = st.text_input("Temp °C",  placeholder="37.2")
        sats = st.text_input("O2 Sats",  placeholder="98%")

# ─────────────────────────────────────────
#  MAIN — TWO TABS
# ─────────────────────────────────────────
tab_text, tab_audio = st.tabs([
    "📝 Type Consultation Note",
    "🎙️ Upload Audio Recording"
])

raw_input   = ""
analyse     = False

# ── Tab 1: Text input ──
with tab_text:
    st.subheader("Consultation Note")
    raw_input_text = st.text_area(
        "Enter the patient presentation",
        placeholder="e.g. 4-month history of fatigue, weight gain, feeling cold all the time, constipation and low mood.",
        height=150,
        key="text_input"
    )
    if st.button("Analyse Case", type="primary",
                 use_container_width=True, key="btn_text"):
        raw_input = raw_input_text
        analyse   = True

# ── Tab 2: Audio upload ──
with tab_audio:
    st.subheader("Upload Consultation Recording")
    st.info(
        "Upload an audio recording of the consultation. "
        "Whisper will transcribe it locally — "
        "no audio data leaves your machine."
    )

    audio_file = st.file_uploader(
        "Choose audio file",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
    )

    whisper_size = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small"],
        index=1,
        help="Base is recommended — good accuracy, runs on modest hardware"
    )

    if audio_file:
        st.audio(audio_file, format=audio_file.type)

        if st.button("Transcribe & Analyse",
                     type="primary",
                     use_container_width=True,
                     key="btn_audio"):

            with st.spinner("Transcribing audio with Whisper..."):
                try:
                    from data_processing.audio_transcriber import AudioTranscriber

                    # save uploaded file to temp location
                    suffix = os.path.splitext(audio_file.name)[1]
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name

                    # transcribe
                    transcriber = AudioTranscriber(model_size=whisper_size)
                    result      = transcriber.process_audio(tmp_path)
                    os.unlink(tmp_path)

                    # show transcript
                    st.success("Transcription complete")
                    st.subheader("Transcript")
                    st.write(result["transcript"])

                    # auto-fill from extracted case data
                    case_data = result["case_data"]

                    st.subheader("Extracted Case Data")
                    col_a, col_b = st.columns(2)

                    with col_a:
                        if case_data["age"]:
                            st.metric("Age detected",  case_data["age"])
                        st.metric("Sex detected",  case_data["sex"])
                        if case_data["pmh"]:
                            st.write("**PMH:**", ", ".join(case_data["pmh"]))
                        if case_data["medications"]:
                            st.write("**Medications:**",
                                     ", ".join(case_data["medications"]))

                    with col_b:
                        if case_data["vitals"]:
                            for k, v in case_data["vitals"].items():
                                st.metric(k.upper(), v)

                    # set values for pipeline
                    raw_input = result["transcript"]
                    analyse   = True

                except Exception as e:
                    st.error(f"Transcription error: {str(e)}")

# ─────────────────────────────────────────
#  RUN PIPELINE — SHARED BY BOTH TABS
# ─────────────────────────────────────────
if analyse and raw_input.strip():
    pmh         = [p.strip() for p in pmh_input.split("\n") if p.strip()]
    medications = [m.strip() for m in med_input.split("\n")  if m.strip()]
    vitals      = {}
    if hr:   vitals["hr"]   = hr
    if bp:   vitals["bp"]   = bp
    if temp: vitals["temp"] = temp
    if sats: vitals["sats"] = sats

    with st.spinner("Running clinical AI pipeline..."):
        try:
            orch   = DiagnosticOrchestrator()
            report = orch.run(
                raw_input=raw_input,
                case_id=case_id,
                age=age,
                sex=sex,
                pmh=pmh,
                medications=medications,
                vitals=vitals if vitals else None
            )
            gate   = SafetyGate()
            report = gate.validate(report)

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            st.stop()

    st.divider()

    # ── Urgent alerts ──
    if report.urgent_alerts:
        st.subheader("⚠️ Urgent Alerts")
        for alert in report.urgent_alerts:
            st.error(alert)
    else:
        st.success("No urgent alerts — no cannot-miss diagnoses detected")

    st.divider()

    # ── Three column output ──
    col_diff, col_tests, col_audit = st.columns([2, 2, 1])

    with col_diff:
        st.subheader("Ranked Differentials")
        if report.differentials:
            for i, diff in enumerate(report.differentials, 1):
                confidence_pct = int(diff.confidence * 100)
                with st.expander(
                    f"{i}. {diff.diagnosis} — {confidence_pct}%",
                    expanded=(i == 1)
                ):
                    if confidence_pct > 0:
                        st.progress(diff.confidence)
                    if diff.reasoning:
                        st.write(diff.reasoning)
                    if diff.red_flags:
                        for rf in diff.red_flags:
                            st.warning(rf)
        else:
            st.info("No differentials extracted — try a more detailed note.")

    with col_tests:
        st.subheader("Recommended Investigations")
        if report.test_recommendations:
            for i, test in enumerate(report.test_recommendations, 1):
                priority_color = {
                    "urgent":  "🔴",
                    "soon":    "🟡",
                    "routine": "🟢"
                }.get(test.priority, "⚪")
                st.markdown(f"**{i}. {priority_color} {test.test_name}**")
                if test.rationale:
                    st.caption(test.rationale)
                st.write("")
        else:
            st.info("No investigations extracted.")

    with col_audit:
        st.subheader("Audit Trail")
        for entry in report.audit_trail:
            status = "✅" if entry.success else "❌"
            st.markdown(f"{status} `{entry.step}`")
        st.caption(f"Case: {report.case_id}")
        st.caption(f"Model: qwen:1.8b")

    st.divider()
    st.warning(f"⚠️ {report.disclaimer}")