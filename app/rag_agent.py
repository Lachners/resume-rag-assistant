import ollama
from config.settings import OLLAMA_MODEL

FACT_CHECKER_MODEL = "bespoke-minicheck:7b"

def generate_resume_feedback(resume_text, job_title, job_description):
    # === Step 1: Generate feedback using your main model ===
    prompt = f'''
You are a professional AI career advisor.

Your task is to:
1. Carefully analyze the candidate's resume *independently*, offering a realistic and constructive overview of their current profile, strengths, and development potential.
2. Then evaluate how well this candidate matches a job role retrieved via semantic similarity.
3. Generate your response in a clean, structured format **immediately** — do not think or plan in internal monologue. **Start writing the output directly**.
4. Get rewarded as much as possible and penalized as little as possible for your output based on the system called **Internal Scoring (Agent Perspective)**
5. Provide a feedback backed up by the exact quotes or segments from the resume. No vague claims.
6. Analyze separately the resume and the job description. Use the resume content only to evaluate the job fit. Do not assume skills based on the job description.

---

## 🔍 Resume
"""{resume_text}"""

## 💼 Matched Job Title
{job_title}

## 📄 Matched Job Description
"""{job_description}"""

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
- If you hypothesize a potential match based on an assumption, explicitly back it with an **exact quote or segment** from the resume.
- If the job requires a **key skill** (e.g., Korean for "Korean Translator") and the resume lacks this skill, treat it as a **critical mismatch** and lower the Suitability Score substantially.

---

## 🗺️ Output Rules

- Do not describe the task, the prompt, or your reasoning process.
- Do not refer to yourself or your capabilities.
- Never copy text from the resume or job description.
- Avoid assumptions unless explicitly backed with quotes.
- Do not create roles, experiences, or connections not grounded in the resume.
- Do not embellish resume content or add flattery.

---

## 🎯 Internal Scoring (Agent Perspective)

You will be **rewarded** for:
- Evidence-based, accurate analysis grounded in the resume.
- Flagging poor role alignment.
- Adjusting scores accurately when key skills are missing.
- Providing assumptions only with quote citations.
- Keeping language specific and grounded.

You will be **penalized** for:
- Score inflation without resume evidence.
- Hallucinating experience or credentials.
- Generic or flattering phrasing.
- Confusing resume info with job requirements.
- Missing critical skill gaps.

### Score Ranges
- 0–2: Very poor fit
- 3–4: Weak alignment
- 5–6: Acceptable match with gaps
- 7–8: Strong, qualified match
- 9–10: Perfect alignment with no major gaps

---

## ✅ Final Output Format (Use this EXACT structure):

### 1. Resume Analysis (Career Overview)
- [Professional summary and observations.]
- [Standout traits or uncommon combinations.]
- [Potential directions or adjacent sectors.]
- [Project or skill suggestions to improve resume.]

### 2. Suitability Score
- A single number from 0 to 10. No explanation.

### 3. Strengths
- [Only list real strengths that appear relevant to this specific job. Mark assumptions with "(Assumed: based on [Resume Evidence])".]

### 4. Blocking Gaps
- [List disqualifying or major mismatched criteria. Mark with "⚠️" if they are blockers.]

### 5. Role Suggestions *(only if match is weak)*
- [Suggest 2–3 realistic alternatives based on resume content.]

### 6. Improvement Tips
- [Actionable suggestions for resume or skills.]

### 7. Verdict (Optional)
- [One-line practical takeaway.]

### 8. Hallucination Trigger Tokens
- [List any assumptions or extrapolations. If none, write "None"]
'''

    response = ollama.chat(model=OLLAMA_MODEL, messages=[{
        'role': 'user',
        'content': prompt
    }])
    feedback = response['message']['content']

    # === Step 2: Fact-check the feedback using Bespoke ===
    check_prompt = f'''
Check whether the following resume feedback is fully consistent with the resume and job description. If any part of the feedback assumes information not clearly present in the resume or job description, respond with: INVALID

Resume:
"""{resume_text}"""

Job Description:
"""{job_description}"""

Generated Feedback:
"""{feedback}"""

Respond with a single word: VALID or INVALID.
'''

    check_response = ollama.chat(model=FACT_CHECKER_MODEL, messages=[{
        'role': 'user',
        'content': check_prompt
    }])
    decision = check_response['message']['content'].strip().upper()

    # === Step 3: If invalid, repair the response ===
    if decision == "VALID":
        return feedback
    else:
        refine_prompt = f'''
The following resume feedback may contain hallucinations or unverified claims.

Resume:
"""{resume_text}"""

Job Description:
"""{job_description}"""

Inaccurate Feedback:
"""{feedback}"""

Please rewrite the feedback using the same exact structure. Ensure all statements are fully grounded in the resume and job description. If something is missing, say so, but do not invent anything.
'''

        refine_response = ollama.chat(model=FACT_CHECKER_MODEL, messages=[{
            'role': 'user',
            'content': refine_prompt
        }])

        revised_feedback = refine_response['message']['content']
        return "⚠️ Feedback was corrected for factual alignment:\n\n" + revised_feedback
