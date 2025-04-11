import ollama
from config.settings import OLLAMA_MODEL

FACT_CHECKER_MODEL = "bespoke-minicheck:7b"

def generate_resume_feedback(resume_text, job_title, job_description):
    # === Step 1: Generate feedback using your main model ===
    prompt = f"""
You are a professional AI career advisor.

Your task is to:
1. Carefully analyze the candidate's resume *independently*, offering a realistic and constructive overview of their current profile, strengths, and development potential.
2. Then evaluate how well this candidate matches a job role retrieved via semantic similarity.
3. Generate your response in a clean, structured format **immediately** — do not think or plan in internal monologue. **Start writing the output directly**.

---

## 🔍 Resume
\"\"\"{resume_text}\"\"\"

## 💼 Matched Job Title
{job_title}

## 📄 Matched Job Description
\"\"\"{job_description}\"\"\"

---

## 🔧 Guidelines for Analysis

### Step 1: Resume Analysis
- Identify core technical and soft skills.
- Detect standout traits, uncommon combinations, or experience that differentiates the candidate.
- Recommend relevant sectors, adjacent roles, or complementary skills or credentials.
- Propose project ideas or tools that could strengthen the profile.

### Step 2: Job Fit Evaluation
- Do **not assume** skills based on the job description — only reference resume content.
- Identify disqualifying factors (e.g. missing language proficiency, experience, certifications).
- Comment briefly on experience level if inferable (e.g., entry, mid, senior).
- Maintain factual alignment. Avoid mixing unrelated concepts.

---

## 🧭 Output Rules

- **Do NOT generate any <think> or planning section.**
- Do not describe the task, the prompt, or your reasoning process.
- Do not refer to yourself or your capabilities.
- Never copy text from the resume or job description.
- Do not speculate or make up career history, education, skills, or job details.
- Do not create roles or titles that are not grounded in the resume.

---

## ✅ Final Output Format (Use this EXACT structure):

### 1. Resume Analysis (Career Overview)
- [Professional summary and observations.]
- [Standout traits or uncommon combinations.]
- [Potential directions or adjacent sectors.]
- [Project or skill suggestions.]

### 2. Suitability Score
- A single number from 0 to 10. No explanation.

### 3. Strengths
- [Only list real strengths that appear relevant to this specific job.]

### 4. Blocking Gaps
- [List disqualifying or major mismatched criteria.]

### 5. Role Suggestions *(only if match is weak)*
- [Suggest 2–3 realistic alternatives based on resume content.]

### 6. Improvement Tips
- [Actionable suggestions for resume or skills.]

### 7. Verdict (Optional)
- [One-line practical takeaway.]

Now generate your response using the format above:
"""

    response = ollama.chat(model=OLLAMA_MODEL, messages=[{
        'role': 'user',
        'content': prompt
    }])
    feedback = response['message']['content']

    # === Step 2: Fact-check the feedback using Bespoke ===
    check_prompt = f"""
Check whether the following resume feedback is fully consistent with the resume and job description. If any part of the feedback assumes information not clearly present in the resume or job description, respond with: INVALID

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description}\"\"\"

Generated Feedback:
\"\"\"{feedback}\"\"\"

Respond with a single word: VALID or INVALID.
"""

    check_response = ollama.chat(model=FACT_CHECKER_MODEL, messages=[{
        'role': 'user',
        'content': check_prompt
    }])
    decision = check_response['message']['content'].strip().upper()

    # === Step 3: If invalid, repair the response instead of rejecting it ===
    if "VALID" in decision:
        return feedback
    else:
        refine_prompt = f"""
The following resume feedback may contain hallucinations or unverified claims.

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description}\"\"\"

Inaccurate Feedback:
\"\"\"{feedback}\"\"\"

Please rewrite the feedback using the same exact structure. Ensure all statements are fully grounded in the resume and job description. If something is missing, say so, but do not invent anything.
"""

        refine_response = ollama.chat(model=FACT_CHECKER_MODEL, messages=[{
            'role': 'user',
            'content': refine_prompt
        }])

        revised_feedback = refine_response['message']['content']
        return "⚠️ Feedback was corrected for factual alignment:\n\n" + revised_feedback
