# RAG middleware. Build from the project root:
#   docker build -f service/Dockerfile -t klique-rag-service:1.0.0 .
FROM python:3.11-slim
WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt
COPY rag/ rag/
COPY service/ service/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# The Containerized Application Launcher ignores the image entrypoint and runs
# the launch form's command line, which is this same command.
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
