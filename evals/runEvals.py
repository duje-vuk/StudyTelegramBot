import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import questionGenerator

sampleMaterial = """
Photosynthesis is the process plants use to turn light energy into chemical energy. It happens in the
chloroplasts, which contain the green pigment chlorophyll. The light-dependent reactions take place in the
thylakoid membranes and produce ATP and NADPH, releasing oxygen as a by-product of splitting water.
The Calvin cycle happens in the stroma. It uses ATP and NADPH to fix carbon dioxide into sugars, with the
enzyme RuBisCO doing the carbon fixation step. Factors that limit the rate of photosynthesis include light
intensity, carbon dioxide concentration and temperature.
"""

results = []


def check(name, passed, detail=""):
    results.append(passed)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name} {detail}")


questions = questionGenerator.askGroqForQuestions(sampleMaterial)
check("generates 5-10 questions", 5 <= len(questions) <= 10, f"(got {len(questions)})")
check("every question has a known type", all(question["type"] != "unknown" for question in questions))
check("mentions key facts", any("calvin" in (question["question"] + question["answer"]).lower() for question in questions),
      "(looks for 'Calvin')")

uniqueQuestions = {question["question"].lower() for question in questions}
check("no duplicate questions", len(uniqueQuestions) == len(questions))

topicName = questionGenerator.askGroqForTopicName(sampleMaterial)
wordCount = len(topicName.split())
check("topic name is 2-5 words", 2 <= wordCount <= 5 or topicName == "Photosynthesis", f"(got '{topicName}')")

passedCount = sum(results)
print(f"\n{passedCount}/{len(results)} checks passed")
sys.exit(0 if passedCount == len(results) else 1)
