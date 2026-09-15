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

#include "xla/backends/gpu/runtime/tensor_ir_kernel_thunk.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "tensor_ir/Runtime/Types.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

namespace rt = ::tensor_ir::rt;

TensorIrKernelThunk::TensorIrKernelThunk(
    ThunkInfo thunk_info, rt::IRuntimeKernelPtr kernel,
    const emitters::KernelArguments& kernel_arguments)
    : Thunk(Kind::kTensorIrKernel, std::move(thunk_info)),
      kernel_(std::move(kernel)),
      args_(kernel_arguments.GetArgumentShapedSlices()),
      written_(kernel_arguments.GetArgumentOutputFlags()) {}

std::string TensorIrKernelThunk::ToString(int indent) const {
  const std::string indent_str(indent * 2, ' ');
  return absl::StrCat(indent_str, "TensorIrKernelThunk(", kernel_->name(),
                      ")");
}

absl::Status TensorIrKernelThunk::Initialize(const InitializeParams& params) {
  rt::Status status = kernel_->initializeRuntimeState();
  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to initialize TensorIR kernel ", kernel_->name(),
                     ": ", status.message()));
  }
  return absl::OkStatus();
}

Thunk::BufferUses TensorIrKernelThunk::buffer_uses() const {
  BufferUses uses;
  uses.reserve(args_.size());
  for (int i = 0; i < args_.size(); ++i) {
    uses.push_back(written_[i]
                       ? BufferUse::Write(args_[i].slice, args_[i].shape)
                       : BufferUse::Read(args_[i].slice, args_[i].shape));
  }
  return uses;
}

absl::Status TensorIrKernelThunk::ExecuteOnStream(const ExecuteParams& params) {
  // Marshal the buffer arguments into the TVM-FFI-style calling convention
  // expected by `IRuntimeKernel::launch`.
  //
  // TODO: this only threads through raw device pointers. Kernels compiled
  // with a non-uniform kernel argument layout (dynamic shapes/strides
  // encoded as extra scalar arguments) aren't supported.
  std::vector<rt::Any> packed_args;
  packed_args.reserve(args_.size());
  for (const ShapedSlice& arg : args_) {
    se::DeviceAddressBase buf =
        params.buffer_allocations->GetDeviceAddress(arg.slice);
    packed_args.emplace_back(buf.opaque());
  }
  rt::PackedArgs args(packed_args.data(),
                      static_cast<int32_t>(packed_args.size()));

  rt::Status check_status = kernel_->checkSupport(args);
  if (!check_status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIR kernel ", kernel_->name(),
                     " doesn't support the given arguments: ",
                     check_status.message()));
  }

  size_t workspace_size = kernel_->queryWorkspaceSize(args);
  if (workspace_size != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "TensorIR kernel ", kernel_->name(),
        " requires a non-zero workspace, which isn't supported yet"));
  }
  rt::Workspace workspace;

  rt::Stream stream = static_cast<rt::Stream>(
      params.stream->platform_specific_handle().stream);

  rt::Status launch_status = kernel_->launch(args, workspace, stream);
  if (!launch_status.ok()) {
    return absl::InternalError(absl::StrCat("Failed to launch TensorIR kernel ",
                                            kernel_->name(), ": ",
                                            launch_status.message()));
  }
  return absl::OkStatus();
}

absl::StatusOr<ThunkProto> TensorIrKernelThunk::ToProto() const {
  return absl::UnimplementedError(
      "TensorIrKernelThunk::ToProto is not implemented");
}

}  // namespace xla::gpu
