FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN useradd -m -u 1000 apiuser

COPY --from=builder /root/.local /home/apiuser/.local

ENV PATH=/home/apiuser/.local/bin:$PATH

COPY --chown=apiuser:apiuser app/ ./app/
COPY --chown=apiuser:apiuser model/ ./model/

USER apiuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
