FROM python:3.11-slim

WORKDIR /app

RUN useradd -m user

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/processed reports mlruns && \
    chown -R user:user /app

USER user

CMD ["python", "-m", "src.pipeline"]
