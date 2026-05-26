FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    accelerate \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    jupyter \
    pyyaml \
    fastapi \
    uvicorn

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]