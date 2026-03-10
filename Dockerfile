ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install clang-20 and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates curl make && \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | \
        tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc > /dev/null && \
    echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" \
        > /etc/apt/sources.list.d/llvm-20.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends clang-20 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy tt-metal submodule first (changes rarely — large cached layer)
COPY third_party/ third_party/

# Build tt-metal SDK from submodule
RUN cd third_party/tt-metal && ./build_metal.sh

# Copy project source and build
COPY src/ src/
COPY Makefile .
RUN make -j$(nproc)

# Model is downloaded at runtime via the binary's built-in HuggingFace resolver.
# To bake the model into the image for faster cold starts, uncomment:
# RUN mkdir -p /models && \
#     curl -L -o /models/Qwen3.5-9B-BF16.gguf \
#       "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-BF16.gguf"
# ENV MODEL_PATH=/models/Qwen3.5-9B-BF16.gguf

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODEL_PATH=unsloth/Qwen3.5-9B-GGUF:BF16
ENV TT_METAL_RUNTIME_ROOT=/app/third_party/tt-metal
ENV QUIET=1
ENV PORT=8888

EXPOSE 8888

CMD ["/entrypoint.sh"]
