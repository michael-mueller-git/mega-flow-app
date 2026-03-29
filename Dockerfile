FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    ffmpeg \
    libgl1 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install git+https://github.com/cvg/megaflow.git
RUN git clone https://github.com/cvg/megaflow.git /app

WORKDIR /app

RUN pip install /app

# Download model
RUN python3 -c 'from megaflow import MegaFlow; MegaFlow.from_pretrained("megaflow-flow")' || true
RUN python3 -c 'from megaflow import MegaFlow; MegaFlow.from_pretrained("megaflow-track")' || true

# Belolw does cause runtime error
# RUN pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch290/index.html
# RUN pip install xformers

COPY ./demo_gradio.py /app/demo_gradio.py

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GRADIO_ANALYTICS_ENABLED="false"
ENV DISABLE_TELEMETRY="1"
ENV HF_HUB_DISABLE_TELEMETRY="1"
ENV DO_NOT_TRACK="1"
ENV HF_HUB_OFFLINE="1"
ENV TRANSFORMERS_OFFLINE="1"

CMD ["python", "demo_gradio.py"]

