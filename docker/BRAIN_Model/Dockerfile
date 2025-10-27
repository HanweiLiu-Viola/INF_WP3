# ---- INF_WP3 :: BRAIN_Model (placeholder) ----
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . /workspace
CMD ["bash", "-lc", "echo BRAIN_Model placeholder; sleep infinity"]
