# Python versiyonu 3.12 ToDo
FROM python:3.12-slim

# Ortam ayarları
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Çalışma dizini:
WORKDIR /app

# Sistem kütüphaneleri
RUN apt-get update && apt-get install -y gcc libpq-dev git

# Gerekli & Proje dosyalarının kopyası:
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

# PYTHONPATH ve proje kökü
ENV PYTHONPATH=/app/src
ENV PROJECT_ROOT=/app/

EXPOSE 8000

# FastAPI servis :
CMD ["uvicorn", "algorithms.main:app", "--port", "8000",  "--reload"]