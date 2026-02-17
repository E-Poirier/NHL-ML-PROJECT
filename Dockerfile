# NHL Point Prediction API (Section 10)
# Build after running pipeline: models/ and data/features/ must exist
FROM python:3.11-slim

WORKDIR /app

# API dependencies only (keeps image smaller)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# App code and artifacts
COPY api/ api/
COPY config/ config/
COPY model_registry.json .
COPY models/ models/
COPY data/features/ data/features/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
