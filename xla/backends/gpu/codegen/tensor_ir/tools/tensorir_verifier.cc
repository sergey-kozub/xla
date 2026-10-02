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

// Verifies HLO-to-TensorIR lowering by comparing a CudaTile execution with
// the result produced by XLA's HLO evaluator.

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "tensor_ir/Compiler/CompileOptions.h"
#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Utils/ComputeCapability.h"
#include "xla/backends/gpu/codegen/tensor_ir/conversion.h"
#include "xla/backends/gpu/codegen/tensor_ir/support.h"
#include "xla/backends/gpu/runtime/tensor_ir_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/error_spec.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace xla::gpu::tensor_ir {
namespace {

namespace se = ::stream_executor;

using mlir::nv_tensor_ir::ICompiler;
using mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompileOptions;

// Generate a random literal for a given shape.
absl::StatusOr<Literal> MakeRandomLiteral(const Shape& shape,
                                          std::minstd_rand0* engine) {
  return primitive_util::ArrayTypeSwitch(
      [&](auto type) -> absl::StatusOr<Literal> {
        return LiteralUtil::CreateRandomLiteral<type>(shape, engine, 0.0, 1.0);
      },
      shape.element_type());
}

// Runs the compiled TensorIR kernel through XLA's `TensorIrKernelThunk`
// runtime (rather than driving `IRuntimeKernel` directly), so the verifier
// exercises the same execution path used by the GPU backend.
absl::StatusOr<Literal> RunModule(mlir::ModuleOp module, ICompiler* compiler,
                                  const CudaTileCompileOptions& options,
                                  se::StreamExecutor* executor,
                                  se::Stream* stream,
                                  const std::vector<Literal>& inputs,
                                  const Shape& output_shape) {
  // Compile the MLIR module.
  auto kernel_or = compiler->compile(module, options);
  if (!kernel_or.ok()) {
    return absl::InternalError(kernel_or.status().message());
  }
  ::tensor_ir::rt::IRuntimeKernelPtr kernel = std::move(*kernel_or);

  // Allocate device buffers for the inputs and the output, and copy the
  // inputs to the device. `allocations` is reserved upfront since
  // `BufferAllocation::Slice` below holds a pointer into it.
  se::StreamExecutorAddressAllocator allocator(executor);
  int device_ordinal = executor->device_ordinal();

  std::vector<se::ScopedDeviceAddress<uint8_t>> device_buffers;
  std::vector<BufferAllocation> allocations;
  std::vector<emitters::KernelArgument> kernel_arguments;
  device_buffers.reserve(inputs.size() + 1);
  allocations.reserve(inputs.size() + 1);
  kernel_arguments.reserve(inputs.size() + 1);

  for (const Literal& input : inputs) {
    ABSL_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> buffer,
        allocator.Allocate(device_ordinal, input.size_bytes()));
    ABSL_RETURN_IF_ERROR(stream->Memcpy(buffer.ptr(), input.untyped_data(),
                                        input.size_bytes()));
    allocations.emplace_back(allocations.size(), input.size_bytes(),
                             /*color=*/0);
    BufferAllocation::Slice slice(&allocations.back(), 0, input.size_bytes());
    kernel_arguments.emplace_back(input.shape(), slice);
    device_buffers.push_back(std::move(buffer));
  }

  Literal output_literal = Literal::CreateFromShape(output_shape);
  ABSL_ASSIGN_OR_RETURN(
      se::ScopedDeviceAddress<uint8_t> output_buffer,
      allocator.Allocate(device_ordinal, output_literal.size_bytes()));
  allocations.emplace_back(allocations.size(), output_literal.size_bytes(),
                           /*color=*/0);
  BufferAllocation::Slice output_slice(&allocations.back(), 0,
                                       output_literal.size_bytes());
  emitters::KernelArgument output_arg(output_shape, output_slice);
  output_arg.set_written(true);
  kernel_arguments.push_back(std::move(output_arg));
  device_buffers.push_back(std::move(output_buffer));

  std::vector<se::DeviceAddressBase> buffer_addresses;
  buffer_addresses.reserve(device_buffers.size());
  for (const se::ScopedDeviceAddress<uint8_t>& buffer : device_buffers) {
    buffer_addresses.push_back(buffer.cref());
  }
  BufferAllocations buffer_allocations(buffer_addresses, device_ordinal,
                                       &allocator);

  // Run the kernel through the XLA runtime.
  emitters::KernelArguments args(std::move(kernel_arguments));
  TensorIrKernelThunk thunk(Thunk::ThunkInfo(), std::move(kernel), args);

  Thunk::InitializeParams init_params;
  init_params.executor = executor;
  ABSL_RETURN_IF_ERROR(thunk.Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream,
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);
  ABSL_RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  // Copy the result to the host.
  ABSL_RETURN_IF_ERROR(stream->Memcpy(output_literal.untyped_data(),
                                      device_buffers.back().cref(),
                                      output_literal.size_bytes()));
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return output_literal;
}

absl::Status ProcessHlo(absl::string_view hlo_source, int64_t seed, float aabs,
                        float arel, ICompiler* compiler,
                        const CudaTileCompileOptions& options,
                        se::StreamExecutor* executor, se::Stream* stream) {
  // Load the HLO module from the input file.
  ABSL_ASSIGN_OR_RETURN(auto hlo_module,
                        xla::LoadModuleFromData(hlo_source, "hlo"));

  // Get the fusion computation from the HLO module.
  const HloComputation* comp = hlo_module->entry_computation();
  if (auto fusion = DynCast<HloFusionInstruction>(comp->root_instruction());
      fusion != nullptr) {
    comp = fusion->fused_instructions_computation();
  }
  llvm::outs() << "Computation: " << comp->name() << "\n";

  // Check if the fusion computation can be converted to TensorIR.
  if (auto decision = IsSupportedFusionComputation(*comp);
      !decision.IsAllowed()) {
    return absl::InvalidArgumentError(decision.Explain());
  }

  // Create the MLIR module.
  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect,
                      mlir::cuda_tile::CudaTileDialect,
                      mlir::arith::ArithDialect>();

  mlir::Location location = mlir::UnknownLoc::get(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(location);

  // Run the HLO to TensorIR conversion.
  ABSL_ASSIGN_OR_RETURN(auto graph_op, ConvertFusionComputation(*comp, module));
  if (llvm::failed(graph_op.verify())) {
    return absl::InternalError("Invalid TensorIR graph");
  }
  graph_op.walk([&](mlir::Operation* op) {
    op->setLoc(location);
  });

  // Create random inputs.
  std::minstd_rand0 engine(seed);
  std::vector<Literal> inputs;
  for (const HloInstruction* parameter : comp->parameter_instructions()) {
    ABSL_ASSIGN_OR_RETURN(auto input,
                          MakeRandomLiteral(parameter->shape(), &engine));
    inputs.push_back(std::move(input));
  }

  // Get the results and compare.
  ABSL_ASSIGN_OR_RETURN(Literal reference,
                        HloEvaluator().Evaluate(*comp, inputs));
  ABSL_ASSIGN_OR_RETURN(
      Literal result, RunModule(module, compiler, options, executor, stream,
                                inputs, reference.shape()));
  return literal_comparison::Near(reference, result, ErrorSpec(aabs, arel),
                                  std::nullopt, nullptr);
}

absl::Status RealMain(absl::string_view input_file, bool split, int64_t seed,
                      float aabs, float arel, absl::string_view target) {
  // Build the compilation options.
  auto target_sm = mlir::nv_tensor_ir::SmTarget::fromString(target);
  if (llvm::failed(target_sm)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid target SM: ", target));
  }
  CudaTileCompileOptions options(*target_sm);

  // Create the compiler instance.
  auto compiler =
      ICompiler::create(mlir::nv_tensor_ir::CompilerBackend::CudaTile);
  if (!compiler) {
    return absl::InternalError("Failed to create the CudaTile compiler");
  }

  // Set up the XLA StreamExecutor runtime used to execute the kernel.
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                        PlatformUtil::GetDefaultPlatform());
  ABSL_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        platform->ExecutorForDevice(0));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                        executor->CreateStream());

  // Read the input file.
  std::string input_data;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), input_file, &input_data));

  // Split the input file into multiple HLO modules.
  std::vector<absl::string_view> parts =
      split ? absl::StrSplit(input_data, "// -----\n")
            : std::vector<absl::string_view>({input_data});

  // Process HLO modules, print the errors if any.
  int error_count = 0;
  for (absl::string_view hlo_data : parts) {
    auto result = ProcessHlo(hlo_data, seed, aabs, arel, compiler.get(),
                             options, executor, stream.get());
    if (!result.ok()) {
      llvm::errs() << result.message() << "\n";
      ++error_count;
    }
  }
  return error_count > 0
             ? absl::InternalError(absl::StrCat(
                   "Verification failed: ", error_count, " / ", parts.size()))
             : absl::OkStatus();
}

}  // namespace
}  // namespace xla::gpu::tensor_ir

int main(int argc, char** argv) {
  bool split = false;
  int64_t seed = 0;
  float aabs = 1e-3;
  float arel = 1e-3;
  std::string target = "sm_100f";

  std::vector<tsl::Flag> flags = {
      tsl::Flag("split", &split, "Split the input file (MLIR-like)."),
      tsl::Flag("seed", &seed, "Random seed for input generation."),
      tsl::Flag("aabs", &aabs, "Absolute error bound."),
      tsl::Flag("arel", &arel, "Relative error bound."),
      tsl::Flag("target", &target, "CudaTile target SM."),
  };

  const std::string usage = tsl::Flags::Usage(argv[0], flags);
  tsl::port::InitMain(argv[0], &argc, &argv);
  LOG_IF(QFATAL, !tsl::Flags::Parse(&argc, argv, flags)) << usage;

  for (int i = 1; i < argc; ++i) {
    absl::Status status =
        xla::gpu::tensor_ir::RealMain(argv[i], split, seed, aabs, arel, target);
    LOG_IF(QFATAL, !status.ok()) << status;
  }
  return 0;
}
