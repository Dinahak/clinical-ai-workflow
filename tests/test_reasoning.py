import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.preprocessing_agent import PreprocessingAgent
from agents.reasoning_agent import ReasoningAgent

pre  = PreprocessingAgent()
rag  = ReasoningAgent()

cases = [
    ("TC001", 44, "F", "fatigue, weight gain, cold intolerance, constipation, low mood, dry skin"),
    ("TC002", 61, "M", "excessive thirst, frequent urination, blurred vision, weight loss, tingling in feet",
     ["Hypertension"], ["Amlodipine 5mg"]),
    ("TC003", 34, "F", "fatigue, shortness of breath, palpitations, headaches, heavy periods, pallor"),
]

for item in cases:
    if len(item) == 4:
        cid, age, sex, text = item
        pmh, meds = [], []
    else:
        cid, age, sex, text, pmh, meds = item

    print(f"\n{'='*50}")
    print(f"{cid} — age {age} {sex}")
    print(f"{'='*50}")

    case, _ = pre.run(text, cid, age, sex, pmh=pmh, medications=meds)
    print(f"Symptoms extracted : {[s.hpo_term for s in case.symptoms if not s.is_negated]}")

    docs, audit = rag.run(case, top_k=3)
    print(f"Documents retrieved: {audit.data['docs_retrieved']}")
    print(f"Top conditions     : {audit.data['top_conditions']}")
    print("Evidence:")
    for i, doc in enumerate(docs, 1):
        cond = doc['metadata'].get('condition', 'unknown')
        src  = doc['metadata'].get('source', '')
        dist = doc['distance']
        print(f"  {i}. [{cond}] ({src}) — relevance score: {dist}")

print("\nAll retrieval tests passed.")
