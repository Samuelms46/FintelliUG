FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#install the dependencies
COPY requirements.txt .
RUN  pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV PYTHONBUFFERED=1

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
