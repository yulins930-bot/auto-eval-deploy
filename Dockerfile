FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY demo/requirements-demo.txt /app/demo/requirements-demo.txt
RUN pip install --upgrade pip && pip install -r /app/demo/requirements-demo.txt

COPY demo /app/demo
COPY prototype /app/prototype

WORKDIR /app/demo
EXPOSE 8080

CMD ["sh", "-c", "gunicorn -w 2 -k gthread --threads 4 --timeout 180 --bind 0.0.0.0:${PORT:-8080} app:app"]
