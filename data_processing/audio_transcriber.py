# data_processing/audio_transcriber.py

import whisper
import re
import os


class AudioTranscriber:
    """
    Transcribes clinical audio consultations using Whisper.
    Runs fully locally — no internet required.
    Extracts structured case data from the transcript.
    """

    def __init__(self, model_size: str = "base"):
        """
        model_size options:
        - "tiny"   — fastest, least accurate (~1GB RAM)
        - "base"   — good balance (~1GB RAM)
        - "small"  — better accuracy (~2GB RAM)
        - "medium" — best for clinical speech (~5GB RAM)
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper ready.")

        # clinical prompt helps Whisper recognise medical terms
        self.clinical_prompt = (
            "This is a family medicine consultation. "
            "The doctor is describing a patient case including "
            "symptoms, medications, past medical history, "
            "and vital signs."
        )

    # ─────────────────────────────────────────
    #  TRANSCRIBE AUDIO FILE
    # ─────────────────────────────────────────
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file and return the full result.
        Accepts: .mp3 .wav .m4a .ogg .flac
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing: {audio_path}")

        result = self.model.transcribe(
            audio_path,
            language="en",
            initial_prompt=self.clinical_prompt,
            fp16=False  # use fp32 for CPU compatibility
        )

        return {
            "transcript": result["text"].strip(),
            "language":   result["language"],
            "segments":   len(result["segments"])
        }

    # ─────────────────────────────────────────
    #  EXTRACT CASE DATA FROM TRANSCRIPT
    # ─────────────────────────────────────────
    def extract_case_data(self, transcript: str) -> dict:
        """
        Parse the transcript into structured case fields.
        Returns a dict that maps directly to the dashboard form.
        """
        text = transcript.lower()

        return {
            "raw_input":    transcript,
            "age":          self._extract_age(text),
            "sex":          self._extract_sex(text),
            "pmh":          self._extract_pmh(text),
            "medications":  self._extract_medications(text),
            "vitals":       self._extract_vitals(text)
        }

    def _extract_age(self, text: str) -> int:
        """Extract patient age from transcript."""
        patterns = [
            r'(\d+)[- ]year[- ]old',
            r'(\d+)[- ]yo',
            r'age[d]?\s+(\d+)',
            r'patient is (\d+)',
            r'(\d+) year old',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                age = int(match.group(1))
                if 0 < age < 120:
                    return age
        return 0  # unknown

    def _extract_sex(self, text: str) -> str:
        """Extract patient sex from transcript."""
        female_terms = ["female", "woman", "lady", "she ", "her ", "girl"]
        male_terms   = ["male", "man", "gentleman", "he ", "his ", "boy"]

        female_count = sum(1 for t in female_terms if t in text)
        male_count   = sum(1 for t in male_terms   if t in text)

        if female_count > male_count:
            return "F"
        elif male_count > female_count:
            return "M"
        return "F"  # default

    def _extract_pmh(self, text: str) -> list[str]:
        """Extract past medical history mentions."""
        conditions = [
            "hypertension", "diabetes", "asthma", "copd",
            "hypothyroidism", "hyperthyroidism", "depression",
            "anxiety", "atrial fibrillation", "heart failure",
            "chronic kidney disease", "ckd", "epilepsy",
            "rheumatoid arthritis", "osteoporosis", "gout",
            "hypercholesterolaemia", "obesity", "anaemia",
            "cancer", "stroke", "heart attack", "angina",
            "dementia", "parkinson", "multiple sclerosis"
        ]
        found = []
        for condition in conditions:
            if condition in text:
                found.append(condition.title())
        return found

    def _extract_medications(self, text: str) -> list[str]:
        """Extract medication mentions from transcript."""
        medications = [
            "amlodipine", "lisinopril", "ramipril", "bisoprolol",
            "atorvastatin", "simvastatin", "metformin", "insulin",
            "levothyroxine", "omeprazole", "lansoprazole",
            "aspirin", "clopidogrel", "warfarin", "apixaban",
            "salbutamol", "seretide", "fostair", "spiriva",
            "sertraline", "fluoxetine", "amitriptyline",
            "paracetamol", "ibuprofen", "naproxen", "codeine",
            "furosemide", "spironolactone", "bendroflumethiazide",
            "prednisolone", "methotrexate", "hydroxychloroquine"
        ]
        found = []
        for med in medications:
            if med in text:
                # try to find dose next to medication name
                pattern = rf"{med}\s*(\d+\s*mg)?"
                match = re.search(pattern, text)
                if match and match.group(1):
                    found.append(f"{med.title()} {match.group(1)}")
                else:
                    found.append(med.title())
        return found

    def _extract_vitals(self, text: str) -> dict:
        """Extract vital signs from transcript."""
        vitals = {}

        # heart rate
        hr_match = re.search(
            r'(?:heart rate|hr|pulse)[^\d]*(\d{2,3})', text
        )
        if hr_match:
            vitals["hr"] = hr_match.group(1)

        # blood pressure
        bp_match = re.search(
            r'(?:blood pressure|bp)[^\d]*(\d{2,3})[/\\](\d{2,3})', text
        )
        if bp_match:
            vitals["bp"] = f"{bp_match.group(1)}/{bp_match.group(2)}"

        # temperature
        temp_match = re.search(
            r'(?:temperature|temp)[^\d]*(\d{2}\.?\d?)', text
        )
        if temp_match:
            vitals["temp"] = temp_match.group(1)

        # oxygen saturation
        sats_match = re.search(
            r'(?:sats|saturation|o2|spo2)[^\d]*(\d{2,3})', text
        )
        if sats_match:
            vitals["sats"] = f"{sats_match.group(1)}%"

        return vitals

    # ─────────────────────────────────────────
    #  FULL PIPELINE — AUDIO TO CASE DATA
    # ─────────────────────────────────────────
    def process_audio(self, audio_path: str) -> dict:
        """
        Full pipeline:
        Audio file → transcript → structured case data
        """
        transcription = self.transcribe(audio_path)
        case_data     = self.extract_case_data(transcription["transcript"])

        return {
            "transcript":  transcription["transcript"],
            "language":    transcription["language"],
            "segments":    transcription["segments"],
            "case_data":   case_data
        }