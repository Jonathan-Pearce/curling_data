# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# opencv-python-headless still needs libgl1 and libglib2.0-0 at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ──────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY conftest.py pyproject.toml ./

# ── Tests (optional — kept so `pytest` works inside the container) ───────────
COPY tests/ ./tests/

# ── Runtime configuration ────────────────────────────────────────────────────
# Makes `from scraping.xxx import ...` and `from ml.xxx import ...` work.
ENV PYTHONPATH=/app/src

# output/ and example raw data/ are intentionally NOT copied into the image;
# mount them at runtime so results persist on the host (see docker-compose.yml).
RUN mkdir -p /app/output /app/output_test "/app/example raw data"

# ── Default command ──────────────────────────────────────────────────────────
CMD ["/bin/bash"]
