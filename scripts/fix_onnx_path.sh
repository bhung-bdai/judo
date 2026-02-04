#!/bin/bash

# Define the target and source
INCLUDE_DIR="$CONDA_PREFIX/include"
ONNX_SUBDIR="$INCLUDE_DIR/onnxruntime"

# Check if headers are in the 'wrong' place and move them if the subdir doesn't exist
if [ -f "$INCLUDE_DIR/onnxruntime_c_api.h" ] && [ ! -d "$ONNX_SUBDIR" ]; then
    echo "Pixi Activation: Moving ONNX headers into subfolder..."
    mkdir -p "$ONNX_SUBDIR"
    # Move headers and common onnx directories into the subfolder
    mv "$INCLUDE_DIR"/onnxruntime*.h "$ONNX_SUBDIR/" 2>/dev/null
    # Move specific ONNX directories if they exist in the root include
    for dir in core session cpu common; do
        if [ -d "$INCLUDE_DIR/$dir" ]; then
            mv "$INCLUDE_DIR/$dir" "$ONNX_SUBDIR/"
        fi
    done
fi
