from dotenv import load_dotenv
import os
import ollama

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
OLLAMA_MODEL = os.getenv("DEEPSEEK_MODEL")
FACT_CHECKER_MODEL = os.getenv("CHECK_MODEL")

ollama_client = ollama.Client(host=OLLAMA_API_URL)

def generate_resume_feedback(resume_text, job_title, job_description):

    prompt = f'''
You are a strict AI career advisor.

Use ONLY the resume below — do not guess or infer beyond it.

Resume:
\"\"\"{resume_text}\"\"\"

Job Title: {job_title}
Job Description:
\"\"\"{job_description}\"\"\"

🎯 TASK:
1. Analyze the resume honestly and constructively.
2. Evaluate the match to the job *without assuming or completing missing info*.
3. Use clear structure and no internal monologue.
4. Consider seniority and experience level to provide feedback.

🚫 HALLUCINATION GUARDRAILS:
- Do NOT assume any skill not in the resume.
- Do NOT use job description to fill resume gaps.
- Do NOT mention benefits, location, company policies, etc.
- Praise or high scores ONLY if clearly justified by resume content.

✅ FORMAT:
1. **Resume Analysis**
2. **Suitability Score** (0–10 only)
3. **Strengths**
4. **Blocking Gaps** (use ⚠️ if critical)
5. **Role Suggestions** (if match is weak)
6. **Improvement Tips**
7. **Verdict (optional)**
8. **Name and Surname** (and contact/location if available)'''

    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[{
        'role': 'user',
        'content': prompt
    }])
    feedback = response['message']['content']

    # === Step 2: Fact-check the feedback using Bespoke ===
    check_prompt = f'''
Check whether the following resume feedback is fully consistent with the resume and job description. If any part of the feedback assumes information not clearly present in the resume or job description, respond with: NO

Resume:
"""{resume_text}"""

Job Description:
"""{job_description}"""

Generated Feedback:
"""{feedback}"""

Respond with a single word: YES or NO.
'''

    check_response = ollama_client.chat(model=FACT_CHECKER_MODEL, messages=[{
        'role': 'user',
        'content': check_prompt
    }])
    decision = check_response['message']['content'].strip().upper()

    # === Step 3: If invalid, repair the response ===
    if decision == "YES":
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

        refine_response = ollama_client.chat(model=FACT_CHECKER_MODEL, messages=[{
            'role': 'user',
            'content': refine_prompt
        }])

        revised_feedback = refine_response['message']['content']
        return "⚠️ Feedback was corrected for factual alignment:\n\n" + revised_feedback