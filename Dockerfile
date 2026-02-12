FROM python:3.10-slim

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenslide0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiamos requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos TODO el proyecto (wsi, seg, web) dentro de /app
COPY . .

# Carpetas de datos
RUN mkdir -p uploads outputs/seg outputs/wsi data/segmentation

EXPOSE 8000

# IMPORTANTE: Ahora lanzamos la app como un módulo desde la raíz
# Esto permite que app.py vea las carpetas 'wsi' y 'seg'
ENV PYTHONPATH=/app
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]