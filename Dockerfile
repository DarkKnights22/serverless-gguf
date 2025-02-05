FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libopenblas-dev \
    python3-dev \
    python3-venv \
    python3-pip \
    nvidia-cuda-toolkit \
    gcc-12 g++-12 \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 12.2 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Verify correct GCC version
RUN gcc --version && g++ --version

# Force use of GCC 12.2
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12
ENV CXXFLAGS="-Wno-error=deprecated-declarations"

# Set CUDA paths
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda"
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12"

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# Manually build llama-cpp-python
RUN git clone --recursive https://github.com/abetlen/llama-cpp-python.git && \
    cd llama-cpp-python && \
    pip install --no-cache-dir .

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
