FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace

COPY /src /workspace

RUN pip install runpod huggingface-hub jsonpickle

ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.4 -DCMAKE_CUDA_ARCHITECTURES=86 -DCUDAToolkit_ROOT=/usr/local/cuda-12.4 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.4/lib64" FORCE_CMAKE=1 pip install git+https://github.com/DarkKnights22/llama-cpp-python.git

# Start the handler
CMD ["python3", "/src/handler.py"]
