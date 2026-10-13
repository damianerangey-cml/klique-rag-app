# Streamlit front end, for local runs with docker compose. On Klique the
# Streamlit catalog app runs ui/app.py from Git and needs no image.
#   docker build -f ui/Dockerfile -t klique-rag-ui:1.0.0 .
FROM python:3.11-slim
WORKDIR /app
COPY ui/requirements.txt ui/requirements.txt
RUN pip install --no-cache-dir -r ui/requirements.txt
COPY ui/ ui/
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
