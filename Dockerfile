FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY server.py .
COPY triggers.json .
COPY mantenimiento.json .
COPY libros.json .

# Módulos (el cerebro completo)
COPY modules/ modules/

# Migraciones
COPY migrations/ migrations/
COPY migrations_prospective/ migrations_prospective/

# Hooks
COPY hooks/ hooks/

RUN mkdir -p data

# Variables de entorno
ENV MCP_TRANSPORT=sse
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "server.py"]
