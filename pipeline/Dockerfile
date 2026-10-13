# Ingestion pipeline code. Used by docker compose for the local "ingest" job,
# and usable as the --docker image of the pipeline's steps on Klique.
#   docker build -f pipeline/Dockerfile -t klique-rag-pipeline:1.0.0 .
FROM python:3.11-slim
WORKDIR /app
COPY pipeline/requirements.txt pipeline/requirements.txt
RUN pip install --no-cache-dir -r pipeline/requirements.txt
COPY rag/ rag/
COPY pipeline/ pipeline/
COPY corpus/ corpus/
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
CMD ["python", "pipeline/rag_ingest_pipeline.py", "--local"]
