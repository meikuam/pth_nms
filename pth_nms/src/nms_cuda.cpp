// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren
// ------------------------------------------------------------------

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>
#include <stdio.h>

#include "cuda/nms_kernel.h"

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) AT_ASSERTM(x.dim() == 4, #x " must have 4 dimensions")

#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Float, #x " must be float Tensor")
#define CHECK_INT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Int, #x " must be int Tensor")
#define CHECK_LONG(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Long, #x " must be long Tensor")


//extern THCState *state;

namespace torch {
int gpu_nms(
    torch::Tensor keep,
    torch::Tensor num_out,
    torch::Tensor boxes,
    float nms_overlap_thresh
) {
    // boxes has to be sorted
    CHECK_INPUT_CPU(keep);      CHECK_LONG(keep);
    CHECK_INPUT_CUDA(boxes);    CHECK_FLOAT(boxes);
    CHECK_INPUT_CPU(num_out);   CHECK_LONG(num_out);

    // Number of ROIs
    int boxes_num = boxes.size(0);
    int boxes_dim = boxes.size(1);

    float* boxes_flat = boxes.data<float>();

    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
    torch::Tensor mask = torch::zeros({boxes_num, col_blocks}, torch::device(torch::kCUDA).dtype(torch::kLong));
    unsigned long* mask_flat = (unsigned long*)mask.data<long>();

    _nms(boxes_num, boxes_flat, mask_flat, nms_overlap_thresh);
    torch::Tensor mask_cpu = mask.cpu();
    unsigned long * mask_cpu_flat = (unsigned long*)mask_cpu.data<long>();

    torch::Tensor remv_cpu = torch::zeros({col_blocks}, torch::kLong);
    unsigned long* remv_cpu_flat = (unsigned long*)remv_cpu.data<long>();

    long * keep_flat = keep.data<long>();
    long num_to_keep = 0;

    int i, j;
    for (i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv_cpu_flat[nblock] & (1ULL << inblock))) {
      keep_flat[num_to_keep++] = i;
      unsigned long *p = &mask_cpu_flat[0] + i * col_blocks;
      for (j = nblock; j < col_blocks; j++) {
        remv_cpu_flat[j] |= p[j];
      }
    }
    }

    long * num_out_flat = num_out.data<long>();
    * num_out_flat = num_to_keep;

//    THLongTensor_free(mask_cpu);
//    THLongTensor_free(remv_cpu);

    return 1;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "nms",
      &torch::gpu_nms,
      "non maximum suppression GPU");
}
