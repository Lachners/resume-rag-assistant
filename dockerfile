FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install && \
    git clone https://github.com/Lachners/resume-rag-assistant.git . && \
    git lfs pull

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501


CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
