set -e -x
/usr/local/cuda-11.8/bin/nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Wed_Sep_21_10:33:58_PDT_2022
# Cuda compilation tools, release 11.8, V11.8.89
# Build cuda_11.8.r11.8/compiler.31833905_0
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install 'transformers[torch]' sentencepiece