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

#ifndef XLA_BACKENDS_GPU_RUNTIME_TENSOR_IR_KERNEL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_TENSOR_IR_KERNEL_THUNK_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/shaped_slice.h"

namespace xla::gpu {

// Executes a kernel compiled by the TensorIR/CudaTile compiler.
//
// Unlike CustomKernelThunk, the compiled artifact is not a plain
// (cubin, kernel name, launch dimensions) tuple: it's an opaque
// `tensor_ir::rt::IRuntimeKernel` that encapsulates its own grid/block
// dimension computation and is invoked through the TVM-FFI-style
// `PackedArgs` calling convention.
class TensorIrKernelThunk : public Thunk {
 public:
  TensorIrKernelThunk(ThunkInfo thunk_info,
                      ::tensor_ir::rt::IRuntimeKernelPtr kernel,
                      const emitters::KernelArguments& kernel_arguments);

  std::string ToString(int indent) const override;

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  BufferUses buffer_uses() const override;

  // TensorIR kernels aren't serializable to a ThunkProto yet: the compiled
  // `IRuntimeKernel` has no persistent representation we can round-trip.
  absl::StatusOr<ThunkProto> ToProto() const override;

 private:
  ::tensor_ir::rt::IRuntimeKernelPtr kernel_;
  std::vector<ShapedSlice> args_;
  std::vector<bool> written_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_TENSOR_IR_KERNEL_THUNK_H_
