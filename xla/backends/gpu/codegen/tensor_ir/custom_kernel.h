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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CUSTOM_KERNEL_H_
#define XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CUSTOM_KERNEL_H_

#include "absl/status/statusor.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"

namespace xla::gpu::tensor_ir {

// Wraps a kernel compiled by the TensorIR CudaTile backend in an XLA
// `CustomKernel`, so that it can be launched by a `CustomKernelThunk`. Going
// through the generic thunk rather than a bespoke one is what lets XLA
// serialize the kernel and record it into a command buffer.
//
// `kernel` must have come from a compiler created with
// `CompilerBackend::CudaTile`, which only ever returns a
// `CudaTileRuntimeKernel`. `num_arguments` is the number of device buffers the
// kernel is launched with: the fusion's inputs followed by its output.
absl::StatusOr<CustomKernel> MakeCustomKernel(
    const ::tensor_ir::rt::IRuntimeKernel& kernel, int num_arguments);

}  // namespace xla::gpu::tensor_ir

#endif  // XLA_BACKENDS_GPU_CODEGEN_TENSOR_IR_CUSTOM_KERNEL_H_
