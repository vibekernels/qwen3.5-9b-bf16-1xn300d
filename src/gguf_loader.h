#pragma once

#include "model.h"
#include <string>

// Load model weights from GGUF file into GPU memory
// Returns a fully initialized Model struct with all weights on GPU
bool load_model(const std::string& path, Model& model);

// Free all GPU memory associated with the model
void free_model(Model& model);
