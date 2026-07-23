FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV APP_ENV=production
ENV CHECKPOINT_BACKEND=sqlite
ENV LANGGRAPH_STRICT_MSGPACK=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm fonts-wqy-zenhei \
    && node --version \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts --no-audit --no-fund

COPY . .

RUN mkdir -p output/diagrams output/memory output/runtime

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=25).read()"

CMD ["uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000"]
