FROM python:3.12-slim

WORKDIR /app

# System deps needed for scipy/scikit-learn compilation on slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure runtime data dirs exist (csv/ is excluded via .dockerignore)
RUN mkdir -p data/csv

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
