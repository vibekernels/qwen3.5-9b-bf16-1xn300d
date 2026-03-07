NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# RTX 5090 = SM 12.0
CUDA_ARCH = -gencode arch=compute_120,code=sm_120

NVCC_FLAGS = $(CUDA_ARCH) -O3 -std=c++17 --use_fast_math -Xcompiler -Wall -Ithird_party
LDFLAGS = -lcudart -lpthread

BUILD_DIR = build
SRC_DIR = src
TEST_DIR = tests

# Source files (shared between CLI and server)
CU_SOURCES = \
    $(SRC_DIR)/sampling.cu \
    $(SRC_DIR)/kernels/rmsnorm.cu \
    $(SRC_DIR)/kernels/embedding.cu \
    $(SRC_DIR)/kernels/rope.cu \
    $(SRC_DIR)/kernels/attention.cu \
    $(SRC_DIR)/kernels/ffn.cu \
    $(SRC_DIR)/kernels/mamba.cu

CPP_SOURCES = \
    $(SRC_DIR)/gguf_loader.cpp \
    $(SRC_DIR)/tokenizer.cpp \
    $(SRC_DIR)/download.cpp

# Object files
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
ENGINE_OBJ = $(BUILD_DIR)/engine.o
SHARED_OBJECTS = $(CU_OBJECTS) $(CPP_OBJECTS) $(ENGINE_OBJ)

# CLI: cli.cu with CLI main()
# Server: server.cu with HTTP server main()
CLI_OBJ = $(BUILD_DIR)/cli.o
SERVER_OBJ = $(BUILD_DIR)/server.o

CLI_TARGET = qwen-inference
SERVER_TARGET = qwen-server

# Test targets
TEST_UTF8 = $(BUILD_DIR)/test_utf8
TEST_TOKENIZER = $(BUILD_DIR)/test_tokenizer
TEST_INFERENCE = $(BUILD_DIR)/test_inference

.PHONY: all clean test test-cpu test-gpu

all: $(CLI_TARGET) $(SERVER_TARGET)

$(CLI_TARGET): $(SHARED_OBJECTS) $(CLI_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

$(SERVER_TARGET): $(SHARED_OBJECTS) $(SERVER_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# Tests
test: test-cpu test-gpu

test-cpu: $(TEST_UTF8) $(TEST_TOKENIZER)
	@echo "=== Running CPU tests ==="
	./$(TEST_UTF8)
	./$(TEST_TOKENIZER)

test-gpu: $(TEST_INFERENCE)
	@echo "=== Running GPU tests (small context) ==="
	TEST_CTX_SIZE=4096 ./$(TEST_INFERENCE)
	@echo ""
	@echo "=== Running GPU tests (full context) ==="
	./$(TEST_INFERENCE)

$(TEST_UTF8): $(TEST_DIR)/test_utf8.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -o $@ $<

$(TEST_TOKENIZER): $(TEST_DIR)/test_tokenizer.cpp $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/download.o
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $(BUILD_DIR)/test_tokenizer.o $(TEST_DIR)/test_tokenizer.cpp
	$(NVCC) $(NVCC_FLAGS) -o $@ $(BUILD_DIR)/test_tokenizer.o $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/download.o $(LDFLAGS)

$(TEST_INFERENCE): $(TEST_DIR)/test_inference.cu $(SHARED_OBJECTS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# CUDA source compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

# C++ source compilation (compiled through nvcc for cuda_bf16.h)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -x cu -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(CLI_TARGET) $(SERVER_TARGET)
