# MODUL 4 - LAB SESI 2 - Langkah 4: Packaging

FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/serve.py .

COPY mlruns/2/models/m-ae065892f80541ef8f75670cd2184778/artifacts/ ./model/

EXPOSE 8080

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]