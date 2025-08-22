FROM python:3.11-slim
WORKDIR /app

# sys deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential rustc cargo \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



# app code last (keeps cache for deps/model)
COPY . .

EXPOSE 8000
CMD ["python", "-u", "handler.py"]