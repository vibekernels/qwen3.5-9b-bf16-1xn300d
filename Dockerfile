ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies (base image already has clang-20, cmake, git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone tt-metal at the pinned submodule commit and build the SDK.
# The base image has runtime libs but no C++ headers, so we build from source.
# A fake git tag provides the version that CMake's packaging step requires.
ARG TT_METAL_COMMIT=967fb02dd27a79ceea8f85dab2e8ee578f0ba82d
RUN git init third_party/tt-metal && \
    cd third_party/tt-metal && \
    git remote add origin https://github.com/tenstorrent/tt-metal.git && \
    git fetch --depth 1 origin $TT_METAL_COMMIT && \
    git checkout FETCH_HEAD && \
    git submodule update --init --recursive && \
    git tag v0.0.0 && \
    ./build_metal.sh

# Copy project source
COPY src/ src/
COPY third_party/json.hpp third_party/json.hpp
COPY Makefile .

# Build the project
RUN make -j$(nproc)

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Model is downloaded at runtime via the binary's built-in HuggingFace resolver.
# To bake the model into the image for faster cold starts, uncomment:
# RUN mkdir -p /models && \
#     curl -L -o /models/Qwen3.5-9B-BF16.gguf \
#       "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-BF16.gguf"
# ENV MODEL_PATH=/models/Qwen3.5-9B-BF16.gguf

ENV MODEL_PATH=unsloth/Qwen3.5-9B-GGUF:BF16
ENV TT_METAL_RUNTIME_ROOT=/app/third_party/tt-metal
ENV QUIET=1
ENV PORT=8888

EXPOSE 8888

CMD ["/entrypoint.sh"]
