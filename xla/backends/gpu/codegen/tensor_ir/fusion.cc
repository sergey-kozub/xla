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

#include "xla/backends/gpu/codegen/tensor_ir/fusion.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Compiler/CompileOptions.h"
#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Utils/ComputeCapability.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/tensor_ir/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/tensor_ir/conversion.h"
#include "xla/backends/gpu/codegen/tensor_ir/support.h"
#include "xla/backends/gpu/runtime/tensor_ir_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla::gpu {

AsyncThunkSequence TensorIrFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  // Verify the fusion is supported.
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                        fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_backend_config.fusion_backend_config();
  if (backend_config.kind() != kTensorIrFusionKind) {
    return absl::InternalError(
        absl::StrCat("TensorIrFusion: unsupported fusion kind: ",
                     backend_config.kind()));
  }
  if (!backend_config.has_tensor_ir_fusion_config()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TensorIrFusion: missing tensor_ir_fusion_config for fusion: ",
        fusion.ToString()));
  }
  const TensorIrFusionConfig& tensor_ir_config =
      backend_config.tensor_ir_fusion_config();

  const HloComputation* computation = fusion.fused_instructions_computation();
  if (auto decision = tensor_ir::IsSupportedFusionComputation(*computation);
      !decision.IsAllowed()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIrFusion: ", decision.Explain()));
  }

  // Create the MLIR context and module.
  BorrowedMlirContext borrowed_context =
      ir_emitter_context.BorrowMlirContext();
  mlir::MLIRContext& context = **borrowed_context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect,
                      mlir::cuda_tile::CudaTileDialect,
                      mlir::arith::ArithDialect>();

  mlir::ModuleOp module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  ABSL_ASSIGN_OR_RETURN(
      mlir::nv_tensor_ir::GraphOp graph_op,
      tensor_ir::ConvertFusionComputation(*computation, module));
  if (llvm::failed(graph_op.verify())) {
    return absl::InternalError(absl::StrCat(
        "TensorIrFusion: invalid TensorIR graph for fusion: ",
        fusion.ToString()));
  }

  // Update the graph with the real alignment values now that we can compute
  // them from buffer assignment.
  ABSL_ASSIGN_OR_RETURN(
      emitters::KernelArguments kernel_arguments,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(),
                                        &fusion));
  const std::vector<emitters::KernelArgument>& kernel_args =
      kernel_arguments.args();
  int64_t num_inputs = computation->num_parameters();
  if (kernel_args.size() != num_inputs + 1) {
    return absl::InternalError(absl::StrCat(
        "TensorIrFusion: expected a single output buffer for fusion: ",
        fusion.ToString()));
  }
  mlir::StringAttr alignment_attr_name = mlir::StringAttr::get(
      &context, mlir::nv_tensor_ir::TensorIRDialect::getAlignmentAttrName());
  for (int64_t i = 0; i < num_inputs; ++i) {
    graph_op.setArgAttr(
        i, alignment_attr_name,
        mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64),
                               kernel_args[i].alignment()));
  }
  graph_op.setResultAttr(
      0, alignment_attr_name,
      mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64),
                             kernel_args[num_inputs].alignment()));

  // Run the conversion pipeline.
  llvm::SmallVector<int32_t> tile_size(tensor_ir_config.tile_size().begin(),
                                       tensor_ir_config.tile_size().end());
  mlir::nv_tensor_ir::TensorToCudaTilePipelineOptions pipeline_options =
      tensor_ir::GetPipelineOptions(
          ir_emitter_context.gpu_device_info().gpu_compute_capability(),
          tile_size, tensor_ir_config.reduction_tile_size());

  mlir::PassManager pass_manager(&context);
  tensor_ir::CreateTensorIrPipeline(&pass_manager, pipeline_options);
  if (llvm::failed(pass_manager.run(module))) {
    return absl::InternalError(
        absl::StrCat("TensorIrFusion: failed to lower TensorIR to CudaTile "
                     "for fusion: ",
                     fusion.ToString()));
  }

  // Compile the kernel.
  using ::mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompileOptions;
  mlir::FailureOr<mlir::nv_tensor_ir::SmTarget> sm_target =
      mlir::nv_tensor_ir::SmTarget::fromCc(pipeline_options.computeCapability);
  if (llvm::failed(sm_target)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "TensorIrFusion: unsupported compute capability: ",
        pipeline_options.computeCapability));
  }
  CudaTileCompileOptions compile_options(*sm_target, pipeline_options.numCTAs,
                                         pipeline_options.numWarps, tile_size);

  std::unique_ptr<mlir::nv_tensor_ir::ICompiler> compiler =
      mlir::nv_tensor_ir::ICompiler::create(
          mlir::nv_tensor_ir::CompilerBackend::CudaTile);
  auto kernel_or = compiler->compile(module, compile_options);
  if (!kernel_or.ok()) {
    return absl::InternalError(
        absl::StrCat("TensorIrFusion: failed to compile fusion ",
                     fusion.name(), ": ", kernel_or.status().message()));
  }

  ThunkSequence thunks;
  thunks.push_back(std::make_unique<TensorIrKernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          &fusion, ir_emitter_context.GetNextThunkId()),
      std::move(kernel_or).value(), kernel_arguments));
  return std::move(thunks);
}

}  // namespace xla::gpu
