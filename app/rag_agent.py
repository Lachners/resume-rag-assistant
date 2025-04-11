import ollama
from config.settings import OLLAMA_MODEL

def generate_resume_feedback(resume_text, job_title, job_description):
    prompt = f"""
You are an AI career advisor.

Resume:
\"\"\"{resume_text}\"\"\"

Matching Job:
Title: {job_title}
Description:
\"\"\"{job_description}\"\"\"

Generate structured feedback:
## Strengths
## Missing Skills
## Role Suggestions
## Improvement Tips
"""

    response = ollama.chat(model=OLLAMA_MODEL, messages=[{
        'role': 'user',
        'content': prompt
    }])
    return response['message']['content']