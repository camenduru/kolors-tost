FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 diffusers==0.28.2 transformers==4.26.1 accelerate==0.27.2 deepspeed==0.8.1 imageio==2.25.1 numpy==1.21.6 omegaconf==2.3.0 sentencepiece==0.1.99 fire cpm_kernels && \
    git clone -b dev https://github.com/camenduru/Kolors /content/Kolors && cd /content/Kolors && pip install -e . && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/scheduler/scheduler_config.json -d /content/Kolors/weights/Kolors/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/text_encoder/config.json -d /content/Kolors/weights/Kolors/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/text_encoder/configuration_chatglm.py -d /content/Kolors/weights/Kolors/text_encoder -o configuration_chatglm.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/text_encoder/modeling_chatglm.py -d /content/Kolors/weights/Kolors/text_encoder -o modeling_chatglm.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/pytorch_model-00001-of-00007.bin -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model-00001-of-00007.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/pytorch_model-00002-of-00007.bin -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model-00002-of-00007.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/pytorch_model-00003-of-00007.bin -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model-00003-of-00007.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/pytorch_model-00004-of-00007.bin -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model-00004-of-00007.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/pytorch_model-00005-of-00007.bin -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model-00005-of-00007.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/pytorch_model-00006-of-00007.bin -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model-00006-of-00007.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/pytorch_model-00007-of-00007.bin -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model-00007-of-00007.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/text_encoder/pytorch_model.bin.index.json -d /content/Kolors/weights/Kolors/text_encoder -o pytorch_model.bin.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/text_encoder/quantization.py -d /content/Kolors/weights/Kolors/text_encoder -o quantization.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/text_encoder/tokenization_chatglm.py -d /content/Kolors/weights/Kolors/text_encoder -o tokenization_chatglm.py && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/tokenizer.model -d /content/Kolors/weights/Kolors/text_encoder -o tokenizer.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/text_encoder/tokenizer_config.json -d /content/Kolors/weights/Kolors/text_encoder -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/text_encoder/vocab.txt -d /content/Kolors/weights/Kolors/text_encoder -o vocab.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/unet/config.json -d /content/Kolors/weights/Kolors/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/unet/diffusion_pytorch_model.safetensors -d /content/Kolors/weights/Kolors/unet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors -d /content/Kolors/weights/Kolors/unet -o diffusion_pytorch_model.fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/raw/main/vae/config.json -d /content/Kolors/weights/Kolors/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/Kolors/weights/Kolors/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kwai-Kolors/Kolors/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors -d /content/Kolors/weights/Kolors/vae -o diffusion_pytorch_model.fp16.safetensors

COPY ./worker_runpod.py /content/Kolors/worker_runpod.py
WORKDIR /content/Kolors
CMD python worker_runpod.py
