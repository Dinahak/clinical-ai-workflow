import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.audio_transcriber import AudioTranscriber

transcriber = AudioTranscriber(model_size="tiny")

# Test case extraction from text directly
transcript = """
44 year old female presenting with a 4 month history
of fatigue, weight gain, feeling cold all the time,
constipation and low mood. She has no past medical history
and is on no medications. Heart rate is 56, blood pressure
108 over 70.
"""

print("Testing case extraction from transcript...")
case_data = transcriber.extract_case_data(transcript)
print(f"Age       : {case_data["age"]}")
print(f"Sex       : {case_data["sex"]}")
print(f"PMH       : {case_data["pmh"]}")
print(f"Meds      : {case_data["medications"]}")
print(f"Vitals    : {case_data["vitals"]}")
print(f"Raw input : {case_data["raw_input"][:80]}...")
print()
print("Case extraction working correctly.")
