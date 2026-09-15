/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/codegen/tensor_ir/compilation_pipeline.h"

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/PassManager.h"
#include "tensor_ir/Compiler/CudaTile/Pipelines.h"
#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu::tensor_ir {

mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions GetPipelineOptions(
    const stream_executor::GpuComputeCapability& gpu_cc,
    llvm::ArrayRef<int32_t> tile_size, int64_t reduction_tile_size) {
  mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions options;
  options.tileSize.assign(tile_size.begin(), tile_size.end());
  options.reductionTileSize = reduction_tile_size;
  // Skip the TileAnalyzer candidate search when an explicit tile shape is
  // already provided (e.g. by the autotuner).
  options.maxCandidates = tile_size.empty() ? 1 : 0;

  if (const auto* cuda_cc = gpu_cc.cuda_compute_capability()) {
    options.computeCapability = cuda_cc->major * 10 + cuda_cc->minor;
  }
  return options;
}

void CreateTensorIrPipeline(
    mlir::OpPassManager* pm,
    const mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions& options) {
  mlir::nv_tensor_ir::buildTensorToCudaTileConversionPipeline(*pm, options);
}

}  // namespace xla::gpu::tensor_ir
