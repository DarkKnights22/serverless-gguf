# Use a RunPod CUDA-enabled base image
FROM runpod/base:0.4.0-cuda11.8.0

# Ensure CUDA runtime libraries are installed
RUN apt-get update && apt-get install -y \
    cuda-command-line-tools-11-8 \
    libcudnn8 \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CMAKE_ARGS="-DGGML_CUDA=on"

# Ensure NVIDIA CUDA libraries are linked correctly
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Copy Python dependencies
COPY builder/requirements.txt /requirements.txt

# Install dependencies
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Add source files
ADD src .

# Verify CUDA before running
RUN nvcc --version && ldconfig -v | grep cuda

# Run the handler script
CMD ["python3.11", "-u", "/handler.py"]
