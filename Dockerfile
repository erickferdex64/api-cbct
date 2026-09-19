# Worker Runpod · DentalSegmentator (nnU-Net v2)
# Misma combinación que ya validaste en Colab: torch 2.11.0 + CUDA 12.8 + nnunetv2 2.8.1.
# Las ruedas de torch traen su propio runtime de CUDA: no hace falta una imagen base de NVIDIA,
# solo que el host tenga driver para CUDA 12.8 (se filtra en el endpoint, ver README).
FROM python:3.12-slim

ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    nnUNet_raw=/models/nnunet/raw \
    nnUNet_preprocessed=/models/nnunet/preprocessed \
    nnUNet_results=/models/nnunet/results \
    DENTALSEG_MODEL_ROOT=/models/nnunet/results

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models/nnunet/raw /models/nnunet/preprocessed /models/nnunet/results

WORKDIR /app

# 1) torch (la capa más pesada, ~4 GB): va primero para que quede en caché entre builds
RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

# 2) resto de dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) pesos dentro de la imagen (~230 MB, con comprobación de md5)
COPY download_weights.py .
RUN python download_weights.py

# 4) el handler, y una prueba de humo: carga el modelo en CPU. Si nnunetv2/torch/pesos no
#    encajan, el fallo aparece AQUÍ, en el build, y no en el primer job de pago.
COPY handler.py .
RUN python -c "import handler; handler.get_predictor()"

CMD ["python", "-u", "handler.py"]
