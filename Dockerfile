FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY config ./config
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 prediction \
    && mkdir -p /app/artifacts /app/reports /app/data \
    && chown -R prediction:prediction /app
USER prediction

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"
CMD ["python", "scripts/server.py"]
