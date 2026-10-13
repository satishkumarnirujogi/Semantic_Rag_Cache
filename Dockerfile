FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# The lockfile pins the CPU-only PyTorch build, which keeps the image ~2 GB smaller than CUDA
COPY requirements.lock .
RUN pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY config ./config
COPY src ./src
RUN pip install --no-deps -e .

EXPOSE 8000 8501
CMD ["uvicorn", "semantic_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
