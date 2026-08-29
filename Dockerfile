# Preflight — AI governance sidecar. Slim production image.
# The optional model deps (torch/transformers/sentence-transformers) are NOT
# installed here: the app lazy-loads them only when PREFLIGHT_NLI_MODEL /
# PREFLIGHT_EMBED_MODEL / PREFLIGHT_INJECTION_MODEL are set, so the default
# deploy runs the documented lexical fallbacks and the Showdown / benchmark use
# the cached results committed under data/benchmark/. Keeps the image small.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Hosts (Render/Railway/Fly) inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn preflight.proxy:app --host 0.0.0.0 --port ${PORT:-8000}"]
