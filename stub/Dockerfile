# Deterministic stand-in for the embedding and chat endpoints (local runs only).
FROM python:3.11-slim
WORKDIR /app
COPY stub/requirements.txt stub/requirements.txt
RUN pip install --no-cache-dir -r stub/requirements.txt
COPY stub/ stub/
ENV PYTHONUNBUFFERED=1
EXPOSE 9000
CMD ["uvicorn", "stub.openai_stub:app", "--host", "0.0.0.0", "--port", "9000"]
