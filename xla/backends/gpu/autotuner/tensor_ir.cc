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

#include "xla/backends/gpu/autotuner/tensor_ir.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "tensor_ir/Conversion/TensorToCudaTile/Options.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/ComputeCapability.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/codegen/tensor_ir/conversion.h"
#include "xla/backends/gpu/codegen/tensor_ir/support.h"
#include "xla/backends/gpu/codegen/tensor_ir/temp/Arch.h"
#include "xla/backends/gpu/codegen/tensor_ir/temp/Enumerate.h"
#include "xla/backends/gpu/codegen/tensor_ir/temp/Evaluate.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::nv_tensor_ir::tiling_analysis::TilingConfig;

std::unique_ptr<BackendConfig> Pack(const TilingConfig& tiling_config) {
  auto config = std::make_unique<BackendConfig>();
  TensorIrFusionConfig* tensor_ir_config = config->mutable_tensor_ir();
  for (int64_t dim : tiling_config.tileShape) {
    tensor_ir_config->add_tile_size(static_cast<int32_t>(dim));
  }
  tensor_ir_config->set_reduction_tile_size(tiling_config.reductionTileSize);
  return config;
}

// Enumerates up to `num_tilings_to_generate` candidate tilings for `graph`
// and returns the `num_tilings_to_select` best-scoring ones, ranked by an
// estimated memory-pressure score (lower is better).
absl::StatusOr<llvm::SmallVector<TilingConfig>> SelectBestTilings(
    mlir::nv_tensor_ir::GraphOp graph, int64_t num_tilings_to_generate,
    int64_t num_tilings_to_select,
    const mlir::nv_tensor_ir::tiling_analysis::ArchInfo& arch_info) {
  mlir::FailureOr<llvm::SmallVector<TilingConfig>> tilings =
      mlir::nv_tensor_ir::tiling_analysis::enumerateTilings(
          graph, num_tilings_to_generate);
  if (llvm::failed(tilings)) {
    return absl::InvalidArgumentError(
        "TensorIrBackend: failed to enumerate tilings");
  }

  // Simple heuristic: minimize the memory pressure.
  mlir::nv_tensor_ir::tiling_analysis::TilingEvaluator evaluator(arch_info);
  auto score =
      [](const mlir::nv_tensor_ir::tiling_analysis::TilingEvaluation& eval) {
        return eval.blockCount *
               (eval.memoryAccessCacheLines + eval.memoryAccessCount);
      };

  // Score the tilings and keep the best ones.
  llvm::SmallVector<std::pair<TilingConfig, int64_t>> scored;
  scored.reserve(tilings->size());
  for (const TilingConfig& tiling_config : *tilings) {
    auto eval = evaluator.evaluate(graph, tiling_config);
    if (eval.errorStatus == mlir::nv_tensor_ir::tiling_analysis::
                                TilingEvaluation::EvalError::Success) {
      scored.emplace_back(tiling_config, score(eval));
    }
  }
  llvm::sort(scored,
             [](const auto& a, const auto& b) { return a.second < b.second; });
  if (static_cast<int64_t>(scored.size()) > num_tilings_to_select) {
    scored.resize(num_tilings_to_select);
  }

  // Return the best tilings.
  llvm::SmallVector<TilingConfig> result;
  result.reserve(scored.size());
  for (auto& [tiling_config, unused_score] : scored) {
    result.push_back(std::move(tiling_config));
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
TensorIrBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  const auto* fusion = Cast<HloFusionInstruction>(&instr);
  const HloComputation& computation =
      *fusion->fused_instructions_computation();

  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect,
                      mlir::arith::ArithDialect>();
  mlir::ModuleOp module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  ABSL_ASSIGN_OR_RETURN(
      mlir::nv_tensor_ir::GraphOp graph_op,
      tensor_ir::ConvertFusionComputation(computation, module));

  // `enumerateTilings` reads the "iteration_space" attribute that the
  // layout-propagation passes attach to the graph; run them first.
  mlir::PassManager pass_manager(&context);
  pass_manager.addNestedPass<mlir::nv_tensor_ir::GraphOp>(
      mlir::nv_tensor_ir::createLayoutPropagationAnnotationPass());
  pass_manager.addNestedPass<mlir::nv_tensor_ir::GraphOp>(
      mlir::nv_tensor_ir::createLayoutPropagationNormalizationPass());
  if (llvm::failed(pass_manager.run(module))) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIrBackend: layout propagation failed for "
                     "fusion: ",
                     instr.ToString()));
  }

  // Get the compute capability and target-specific architecture information.
  const auto* cuda_cc = target_config()
                            .device_description.gpu_compute_capability()
                            .cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return absl::InvalidArgumentError(
        "TensorIrBackend: requires a CUDA compute capability");
  }
  mlir::FailureOr<mlir::nv_tensor_ir::SmTarget> sm_target =
      mlir::nv_tensor_ir::SmTarget::fromCc(cuda_cc->major * 10 +
                                           cuda_cc->minor);
  if (llvm::failed(sm_target)) {
    return absl::InvalidArgumentError(
        "TensorIrBackend: unsupported compute capability");
  }
  const auto& arch_info = mlir::nv_tensor_ir::tiling_analysis::getArchInfo(
      sm_target->getComputeCapability());

  // Select the best tilings.
  ABSL_ASSIGN_OR_RETURN(
      llvm::SmallVector<TilingConfig> tilings,
      SelectBestTilings(graph_op, /*num_tilings_to_generate=*/1000,
                        /*num_tilings_to_select=*/20, arch_info));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(tilings.size());
  for (const TilingConfig& tiling_config : tilings) {
    configs.push_back(Pack(tiling_config));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
TensorIrBackend::GetDefaultConfig(const HloInstruction& instr) {
  ABSL_ASSIGN_OR_RETURN(auto configs, GetSupportedConfigs(instr));
  if (configs.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("TensorIrBackend: no supported tiling configs for "
                     "instruction: ",
                     instr.ToString()));
  }
  return std::move(configs.front());
}

absl::Status TensorIrBackend::ApplyConfig(HloInstruction& instr,
                                          const BackendConfig& config) {
  if (!config.has_tensor_ir()) {
    return absl::InvalidArgumentError("Expected TensorIrFusionConfig.");
  }
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                   instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig& backend_config =
      *gpu_backend_config.mutable_fusion_backend_config();
  backend_config.set_kind(kTensorIrFusionKind);
  *backend_config.mutable_tensor_ir_fusion_config() = config.tensor_ir();
  ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_backend_config)));
  instr.set_fusion_kind(HloInstruction::FusionKind::kCustom);
  return absl::OkStatus();
}

bool TensorIrBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  const auto* fusion = Cast<HloFusionInstruction>(&instr);
  return tensor_ir::IsSupportedFusionComputation(
             *fusion->fused_instructions_computation())
      .IsAllowed();
}

}  // namespace gpu
}  // namespace xla
