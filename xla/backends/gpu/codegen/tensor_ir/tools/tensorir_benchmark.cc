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

// Benchmarks every post-compilation HLO fusion through TensorIR and XLA.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/log_severity.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "tensor_ir/Compiler/CompileOptions.h"
#include "tensor_ir/Compiler/Compiler.h"
#include "tensor_ir/Compiler/CudaTile/CudaTileCompiler.h"
#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "tensor_ir/Runtime/IRuntimeKernel.h"
#include "tensor_ir/Support/Status.h"
#include "tensor_ir/Transform/Passes.h"
#include "tensor_ir/Utils/ComputeCapability.h"
#include "xla/backends/gpu/codegen/tensor_ir/conversion.h"
#include "xla/backends/gpu/codegen/tensor_ir/support.h"
#include "xla/backends/gpu/codegen/tensor_ir/temp/Arch.h"
#include "xla/backends/gpu/codegen/tensor_ir/temp/Enumerate.h"
#include "xla/backends/gpu/codegen/tensor_ir/temp/Evaluate.h"
#include "xla/backends/gpu/runtime/tensor_ir_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/platform_util.h"
#include "xla/service/restricted/hlo_runner_legacy.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"

namespace xla::gpu::tensor_ir {
namespace {

namespace se = ::stream_executor;

using mlir::nv_tensor_ir::backend::cuda_tile::CudaTileCompileOptions;
using mlir::nv_tensor_ir::tiling_analysis::TilingConfig;

struct BenchmarkResult {
  uint64_t time_ns_p10;
  uint64_t time_ns_p50;
  uint64_t time_ns_p90;
};

struct BenchmarkEntry {
  std::string fusion_name;
  BenchmarkResult xla_result;
  BenchmarkResult tir_result;
};

// Mirrors HloOpProfiler's CUDA implementation, while profiling the CudaTile
// launches and XLA runner launches directly.
class CuptiKernelTracer : public profiler::CuptiTraceCollector {
 public:
  CuptiKernelTracer()
      : profiler::CuptiTraceCollector({}),
        cupti_tracer_(profiler::CuptiTracer::GetCuptiTracerSingleton()) {
    CHECK(cupti_tracer_->IsAvailable());
    profiler::CuptiTracerOptions options;
    options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
    if (!cupti_tracer_->Enable(options, this).ok()) {
      LOG(ERROR) << "Failed to enable CUPTI kernel tracer";
    }
  }

  ~CuptiKernelTracer() override {
    if (enabled_) {
      cupti_tracer_->Disable();
    }
  }

  BenchmarkResult ConsumeResult() {
    if (enabled_) {
      cupti_tracer_->Disable();
      enabled_ = false;
    }
    if (kernel_times_ns_.empty()) {
      return {0, 0, 0};
    }
    absl::c_sort(kernel_times_ns_);
    const auto percentile = [&](uint64_t percentile) {
      return kernel_times_ns_[(kernel_times_ns_.size() - 1) * percentile / 100];
    };
    return {percentile(10), percentile(50), percentile(90)};
  }

 private:
  void AddEvent(profiler::CuptiTracerEvent&& event) override {
    if (event.type == profiler::CuptiTracerEventType::Kernel &&
        event.source == profiler::CuptiTracerEventSource::Activity) {
      kernel_times_ns_.push_back(event.end_time_ns - event.start_time_ns);
    }
  }
  void OnEventsDropped(const std::string& reason, uint32_t count) override {
    LOG(ERROR) << "CUPTI dropped " << count << " events: " << reason;
  }
  void Flush() override {}

  profiler::CuptiTracer* cupti_tracer_;
  bool enabled_ = true;
  std::vector<uint64_t> kernel_times_ns_;
};

// Compile and benchmark a TensorIR module through XLA's `TensorIrKernelThunk`
// runtime (rather than driving `IRuntimeKernel` directly), so the benchmark
// exercises the same execution path used by the GPU backend.
absl::StatusOr<BenchmarkResult> RunModule(
    mlir::ModuleOp module, mlir::nv_tensor_ir::ICompiler* compiler,
    const CudaTileCompileOptions& options, se::StreamExecutor* executor,
    se::Stream* stream, const emitters::KernelArguments& kernel_arguments,
    const BufferAllocations& buffer_allocations, int64_t repeats) {
  // Compile the MLIR module.
  auto kernel_or = compiler->compile(module, options);
  if (!kernel_or.ok()) {
    return absl::InternalError(kernel_or.status().message());
  }
  ::tensor_ir::rt::IRuntimeKernelPtr kernel = std::move(*kernel_or);

  // Run the kernel through the XLA runtime.
  TensorIrKernelThunk thunk(Thunk::ThunkInfo(), std::move(kernel),
                            kernel_arguments);

  Thunk::InitializeParams init_params;
  init_params.executor = executor;
  ABSL_RETURN_IF_ERROR(thunk.Initialize(init_params));

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream,
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  // Run the warmup round.
  ABSL_RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  // Run the benchmark (multiple times).
  CuptiKernelTracer tracer;
  for (int i = 0; i < repeats; ++i) {
    ABSL_RETURN_IF_ERROR(thunk.ExecuteOnStream(execute_params));
  }
  ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return tracer.ConsumeResult();
}

// Autotune using all possible tile sizes (or a subset).
absl::StatusOr<TilingConfig> TryAutotune(
    mlir::nv_tensor_ir::GraphOp graph_op,
    mlir::nv_tensor_ir::ICompiler* compiler, CudaTileCompileOptions options,
    se::StreamExecutor* executor, se::Stream* stream,
    const emitters::KernelArguments& kernel_arguments,
    const BufferAllocations& buffer_allocations, int64_t autotune,
    int64_t repeats) {
  // Search the tiling space.
  constexpr int64_t kTilingEnumerateSize = 10000;
  auto tilings = mlir::nv_tensor_ir::tiling_analysis::enumerateTilings(
    graph_op, kTilingEnumerateSize);
  if (llvm::failed(tilings)) {
    return absl::InternalError("Failed to enumerate tiling combinations");
  }

  // Calculate tiling score based on the estimated memory pressure.
  auto arch_info = mlir::nv_tensor_ir::tiling_analysis::getArchInfo(
      options.computeCapability.getComputeCapability());
  auto evaluator =
      mlir::nv_tensor_ir::tiling_analysis::TilingEvaluator(arch_info);

  llvm::SmallVector<std::pair<
      TilingConfig, mlir::nv_tensor_ir::tiling_analysis::TilingEvaluation>>
      scored_tilings;
  for (const auto& tile_config : *tilings) {
    auto eval = evaluator.evaluate(graph_op, tile_config);
    if (eval.tileStorageBytes <= 262144) { // 256 KB
      scored_tilings.push_back({std::move(tile_config), std::move(eval)});
    }
  }

  // Prune the tilings based on the score.
  auto score = [&](const auto& eval) {
    return eval.blockCount *
           (eval.memoryAccessCacheLines + eval.memoryAccessCount);
  };
  llvm::sort(scored_tilings, [&](const auto& a, const auto& b) {
    int64_t a_score = score(a.second);
    int64_t b_score = score(b.second);
    if (a_score != b_score) {
      return a_score < b_score;
    }
    for (auto [lhs, rhs] :
         llvm::zip_equal(a.first.tileShape, b.first.tileShape)) {
      if (lhs != rhs) {
        return lhs < rhs;
      }
    }
    return a.first.reductionTileSize < b.first.reductionTileSize;
  });
  if (autotune < scored_tilings.size()) {
    scored_tilings.resize(autotune);
  }

  // Try all the possible tile sizes.
  std::optional<TilingConfig> best_tile_config;
  int64_t best_time_ns = INT64_MAX;
  auto module = graph_op->getParentOfType<mlir::ModuleOp>();

  for (const auto& [tile_config, tile_eval] : scored_tilings) {
    options.tileSize.assign(tile_config.tileShape.begin(),
                            tile_config.tileShape.end());
    options.reductionTileSize = tile_config.reductionTileSize;
    ABSL_ASSIGN_OR_RETURN(
        BenchmarkResult result,
        RunModule(module, compiler, options, executor, stream,
                 kernel_arguments, buffer_allocations, repeats));
    if (result.time_ns_p10 < best_time_ns) {
      best_time_ns = result.time_ns_p10;
      best_tile_config = tile_config;
    }
    VLOG(3) << "  " << tile_config.toString() << ": [p10] "
            << result.time_ns_p10 << " ns, [p50] " << result.time_ns_p50
            << " ns, [p90] " << result.time_ns_p90 << " ns, "
            << tile_eval.toString() << ", score: " << score(tile_eval) / 1000;
  }

  // Return the best tile config.
  VLOG(2) << "Selected tile config: " << best_tile_config->toString();
  return std::move(*best_tile_config);
}

// Benchmark a fusion using the TensorIR runner.
absl::StatusOr<BenchmarkResult> BenchmarkFusionTensorIR(
    const HloComputation* comp, absl::Span<Literal> inputs,
    CudaTileCompileOptions options, se::StreamExecutor* executor,
    se::Stream* stream, int64_t repeats, int64_t autotune,
    std::optional<TilingConfig> tile_config) {
  // Create the MLIR module.
  mlir::MLIRContext context;
  context.loadDialect<mlir::nv_tensor_ir::TensorIRDialect,
                      mlir::cuda_tile::CudaTileDialect,
                      mlir::arith::ArithDialect>();

  mlir::Location location = mlir::UnknownLoc::get(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(location);

  // Run the HLO to TensorIR conversion.
  ABSL_ASSIGN_OR_RETURN(auto graph_op,
                        ConvertFusionComputation(*comp, module));
  if (llvm::failed(graph_op.verify())) {
    return absl::InternalError("Invalid TensorIR graph");
  }
  graph_op.walk([&](mlir::Operation* op) {
    op->setLoc(location);
  });

  // Try running layout propagation.
  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::nv_tensor_ir::GraphOp>(
      mlir::nv_tensor_ir::createLayoutPropagationAnnotationPass());
  pm.addNestedPass<mlir::nv_tensor_ir::GraphOp>(
      mlir::nv_tensor_ir::createLayoutPropagationNormalizationPass());
  if (llvm::failed(pm.run(module))) {
    VLOG(1) << "Layout propagation failed";
    return BenchmarkResult{0, 0, 0};
  }

  auto layout = mlir::nv_tensor_ir::tiling_analysis::getInputLayout(graph_op);
  auto layout_shape = layout.getShape();
  VLOG(2) << "Layout shape: [" << absl::StrJoin(layout_shape, ",") << "]";

  // Create the compiler instance.
  auto compiler = mlir::nv_tensor_ir::ICompiler::create(
      mlir::nv_tensor_ir::CompilerBackend::CudaTile);
  if (!compiler) {
    return absl::InternalError("Failed to create the CudaTile compiler");
  }

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
    emitters::KernelArgument& arg =
        kernel_arguments.emplace_back(input.shape(), slice);
    // `se::StreamExecutorAddressAllocator` allocates via cudaMalloc, which is
    // guaranteed to be at least 256-byte aligned.
    arg.set_alignment(kXlaAllocatedBufferAlignBytes);
    device_buffers.push_back(std::move(buffer));
  }

  const Shape& output_shape = comp->root_instruction()->shape();
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
  output_arg.set_alignment(kXlaAllocatedBufferAlignBytes);
  kernel_arguments.push_back(std::move(output_arg));
  device_buffers.push_back(std::move(output_buffer));

  std::vector<se::DeviceAddressBase> buffer_addresses;
  buffer_addresses.reserve(device_buffers.size());
  for (const se::ScopedDeviceAddress<uint8_t>& buffer : device_buffers) {
    buffer_addresses.push_back(buffer.cref());
  }
  BufferAllocations buffer_allocations(buffer_addresses, device_ordinal,
                                       &allocator);
  emitters::KernelArguments args(std::move(kernel_arguments));

  // Update the graph with the real alignment values now that we can compute
  // them from the allocated buffers.
  const std::vector<emitters::KernelArgument>& kernel_args = args.args();
  int64_t num_inputs = comp->num_parameters();
  if (kernel_args.size() != num_inputs + 1) {
    return absl::InternalError("Expected a single output buffer for fusion");
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

  // Run the autotuning for the program shape.
  if (autotune) {
    ABSL_ASSIGN_OR_RETURN(
        auto autotune_config,
        TryAutotune(graph_op, compiler.get(), options, executor, stream, args,
                    buffer_allocations, autotune, /*repeats=*/3));
    tile_config = std::make_optional(autotune_config);
  }

  // Set the tile shape in the compile options.
  if (tile_config.has_value()) {
    if (tile_config->tileShape.size() == layout_shape.size() + 1) {
      tile_config->reductionTileSize = tile_config->tileShape.back();
      tile_config->tileShape.pop_back();
    }
    options.tileSize.assign(tile_config->tileShape.begin(),
                            tile_config->tileShape.end());
    options.reductionTileSize = tile_config->reductionTileSize;
  }

  // Run the module and return the result.
  ABSL_ASSIGN_OR_RETURN(
      BenchmarkResult result,
      RunModule(module, compiler.get(), options, executor, stream, args,
               buffer_allocations, repeats));
  VLOG(1) << "  TIR result: [p10] " << result.time_ns_p10 << " ns, [p50] "
          << result.time_ns_p50 << " ns, [p90] " << result.time_ns_p90 << " ns";
  return result;
}

// Benchmark a fusion using the XLA runner.
absl::StatusOr<BenchmarkResult> BenchmarkFusionXLA(
    std::unique_ptr<HloModule> module, absl::Span<Literal> inputs,
    int64_t repeats) {
  // Create the XLA executable.
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                        PlatformUtil::GetPlatform("gpu"));
  HloRunnerLegacy runner(platform);
  ABSL_ASSIGN_OR_RETURN(
      auto executable,
      runner.CreateExecutable(std::move(module), /*run_hlo_passes=*/false));

  // Run the warmup round.
  llvm::SmallVector<const Literal*> arguments;
  for (const Literal& literal : inputs) {
    arguments.push_back(&literal);
  }
  ABSL_ASSIGN_OR_RETURN(
      auto warmup, runner.ExecuteWithExecutable(executable.get(), arguments,
                                                /*num_repeats=*/1));
  ABSL_RETURN_IF_ERROR(warmup[0].status());

  // Run the benchmark (multiple times).
  CuptiKernelTracer tracer;
  ABSL_ASSIGN_OR_RETURN(
      auto results,
      runner.ExecuteWithExecutable(executable.get(), arguments, repeats));
  for (const auto& result : results) {
    ABSL_RETURN_IF_ERROR(result.status());
  }

  // Return the result.
  BenchmarkResult result = tracer.ConsumeResult();
  VLOG(1) << "  XLA result: [p10] " << result.time_ns_p10 << " ns, [p50] "
          << result.time_ns_p50 << " ns, [p90] " << result.time_ns_p90 << " ns";
  return result;
}

// Benchmark a fusion using the XLA runner and TensorIR.
absl::StatusOr<BenchmarkEntry> BenchmarkFusion(
    std::unique_ptr<HloModule> module, const CudaTileCompileOptions& options,
    se::StreamExecutor* executor, se::Stream* stream, int64_t repeats,
    int64_t autotune, std::optional<TilingConfig> tile_config) {
  VLOG(1) << "Benchmarking: " << module->name();
  const HloComputation* comp = module->entry_computation()
                                   ->root_instruction()
                                   ->fused_instructions_computation();

  // Create uninitialized inputs.
  std::vector<Literal> inputs;
  for (const HloInstruction* parameter : comp->parameter_instructions()) {
    ABSL_ASSIGN_OR_RETURN(auto input, Literal::Make(parameter->shape()));
    inputs.push_back(std::move(input));
  }

  // Run the benchmark.
  BenchmarkEntry result;
  result.fusion_name = module->name();
  ABSL_ASSIGN_OR_RETURN(
      result.tir_result,
      BenchmarkFusionTensorIR(comp, absl::MakeSpan(inputs), options, executor,
                              stream, repeats, autotune, tile_config));
  ABSL_ASSIGN_OR_RETURN(
      result.xla_result,
      BenchmarkFusionXLA(std::move(module), absl::MakeSpan(inputs), repeats));
  return result;
}

void PrintResults(absl::Span<BenchmarkEntry> results) {
  // Print the table header.
  size_t name_size = 16;
  for (const BenchmarkEntry& result : results) {
    name_size = std::max(name_size, result.fusion_name.size());
  }
  std::stringstream str;
  str << "+" << std::string(name_size + 2, '-')
      << "+---------+---------+---------+---------+---------+---------+\n";
  llvm::outs() << str.str();
  llvm::outs()
      << "| Fusion" << std::string(name_size - 5, ' ')
      << "| XLA p10 | XLA p50 | XLA p90 | TIR p10 | TIR p50 | TIR p90 |\n";
  llvm::outs() << str.str();

  // Print the table body.
  for (const BenchmarkEntry& result : results) {
    size_t pad = name_size - result.fusion_name.size();
    llvm::outs() << "| " << result.fusion_name << std::string(pad, ' ') << " |";
    std::array<uint64_t, 6> times = {
        result.xla_result.time_ns_p10, result.xla_result.time_ns_p50,
        result.xla_result.time_ns_p90, result.tir_result.time_ns_p10,
        result.tir_result.time_ns_p50, result.tir_result.time_ns_p90};
    for (uint64_t time_ns : times) {
      if (time_ns != 0) {
        float time_us = static_cast<float>(time_ns) / 1e3;
        llvm::outs() << llvm::format("%8.1f", time_us) << " |";
      } else {
        llvm::outs() << "     n/a |";
      }
    }
    llvm::outs() << "\n";
  }

  // Print the table footer.
  llvm::outs() << str.str();
}

absl::Status BenchmarkCompiledHLO(absl::string_view input_file,
                                  const CudaTileCompileOptions& options,
                                  se::StreamExecutor* executor,
                                  se::Stream* stream, int64_t repeats,
                                  int64_t autotune,
                                  std::optional<TilingConfig> tile_config) {
  // Read the input file.
  ABSL_ASSIGN_OR_RETURN(auto hlo_module,
                        LoadModuleFromFile(std::string(input_file), "hlo"));

  // Run the benchmark and print the results.
  ABSL_ASSIGN_OR_RETURN(
      BenchmarkEntry result,
      BenchmarkFusion(std::move(hlo_module), options, executor, stream,
                      repeats, autotune, tile_config));
  PrintResults(absl::MakeSpan(&result, 1));
  return absl::OkStatus();
}

// Use a no-op custom call target for SPMD partitioning.
void NoOpCustomSPMDPartitioning(void* /*stream*/, void** /*buffers*/,
                                const char* /*opaque*/, int /*opaque_len*/,
                                XlaCustomCallStatus* /*status*/) {}
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("CustomSPMDPartitioning",
                                         NoOpCustomSPMDPartitioning, "CUDA");

absl::Status BenchmarkFullHLO(absl::string_view input_file,
                              const CudaTileCompileOptions& options,
                              se::StreamExecutor* executor, se::Stream* stream,
                              int64_t repeats, int64_t autotune,
                              absl::string_view output_path) {
  // Read the input file.
  ABSL_ASSIGN_OR_RETURN(auto hlo_module,
                        LoadModuleFromFile(std::string(input_file), "hlo"));
  hlo_module->mutable_config().set_use_shardy_partitioner(true);

  // Compile the module to get the list of fusions.
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                        PlatformUtil::GetPlatform("gpu"));
  HloRunnerLegacy runner(platform);
  ABSL_ASSIGN_OR_RETURN(auto executable,
                        runner.CreateExecutable(std::move(hlo_module),
                                                /*run_hlo_passes=*/true));
  ABSL_ASSIGN_OR_RETURN(const HloModule* compiled_module,
                        runner.HloModuleFromWrapped(executable.get()));

  // Create the output directory if needed.
  auto env = tsl::Env::Default();
  if (!output_path.empty()) {
    ABSL_RETURN_IF_ERROR(env->RecursivelyCreateDir(output_path));
  }

  // Deduplicate and collect the list of fusions.
  llvm::SmallVector<const HloFusionInstruction*> fusions;
  absl::flat_hash_set<std::string> fusions_seen;
  int duplicate_count = 0;
  HloPrintOptions print_options = HloPrintOptions::Canonical();
  print_options.set_print_metadata(false);

  for (const HloInstruction* instr :
       compiled_module->entry_computation()->instructions()) {
    if (const auto* fusion = DynCast<HloFusionInstruction>(instr);
        fusion != nullptr) {
      const HloComputation* comp = fusion->fused_instructions_computation();
      std::string key = comp->ToString(print_options);
      key.erase(0, key.find('{'));  // Remove the computation name.
      if (fusions_seen.insert(key).second) {
        fusions.push_back(fusion);
      } else {
        ++duplicate_count;
      }
    }
  }
  if (duplicate_count > 0) {
    llvm::errs() << "Duplicate fusions: " << duplicate_count
                 << ", remaining: " << fusions.size() << "\n";
  }

  // Extract fusions into separate modules.
  llvm::SmallVector<std::unique_ptr<HloModule>> modules;
  absl::flat_hash_map<std::string, llvm::SmallVector<std::string>> unsupported;
  for (const HloFusionInstruction* fusion : fusions) {
    auto module = ExtractInstructionIntoNewModule(*fusion);
    auto decision =
        IsSupportedFusionComputation(*fusion->fused_instructions_computation());

    if (!output_path.empty()) {
      std::string filename = absl::StrCat(
          fusion->name(), decision.IsAllowed() ? "" : "_unsupported", ".hlo");
      ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(
          env, tsl::io::JoinPath(output_path, filename), module->ToString()));
    }

    if (decision.IsAllowed()) {
      modules.push_back(std::move(module));
    } else {
      unsupported[decision.Explain()].push_back(std::string(fusion->name()));
    }
  }

  // Print the information about unsupported fusions.
  if (!unsupported.empty()) {
    for (const auto& [reason, names] : unsupported) {
      llvm::errs() << "[" << names.size() << "] " << reason << " @";
      for (const auto& name : names) {
        llvm::errs() << " " << name;
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "Unsupported fusions: " << (fusions.size() - modules.size())
                 << ", remaining: " << modules.size() << "\n";
  }

  // Run the benchmarks and collect the results.
  llvm::SmallVector<BenchmarkEntry> results;
  for (auto& module : modules) {
    auto result = BenchmarkFusion(std::move(module), options, executor, stream,
                                  repeats, autotune, std::nullopt);
    if (result.ok()) {
      results.push_back(std::move(*result));
    } else {
      llvm::errs() << result.status().message() << "\n";
    }
  }

  // Save the results to a CSV file.
  if (!output_path.empty()) {
    std::stringstream str;
    str << "fusion,xla_p10,xla_p50,xla_p90,tir_p10,tir_p50,tir_p90\n";
    auto to_us = [](uint64_t time_ns) { return float(time_ns) / 1e3; };
    for (const auto& result : results) {
      str << result.fusion_name << ",";
      str << to_us(result.xla_result.time_ns_p10) << ","
          << to_us(result.xla_result.time_ns_p50) << ","
          << to_us(result.xla_result.time_ns_p90) << ","
          << to_us(result.tir_result.time_ns_p10) << ","
          << to_us(result.tir_result.time_ns_p50) << ","
          << to_us(result.tir_result.time_ns_p90) << "\n";
    }
    ABSL_RETURN_IF_ERROR(tsl::WriteStringToFile(
        env, tsl::io::JoinPath(output_path, "results.csv"), str.str()));
  }

  // Print the results.
  PrintResults(absl::MakeSpan(results));
  return absl::OkStatus();
}

absl::Status RealMain(absl::string_view input_file, absl::string_view target,
                      int64_t repeats, int64_t autotune, bool single,
                      absl::string_view tile_shape,
                      absl::string_view output_path) {
  // Build the compilation options.
  auto target_sm = mlir::nv_tensor_ir::SmTarget::fromString(target);
  if (llvm::failed(target_sm)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid target SM: ", target));
  }
  CudaTileCompileOptions options(*target_sm);

  // Parse the tile shape, if provided.
  std::optional<TilingConfig> tile_config;
  if (!tile_shape.empty()) {
    llvm::SmallVector<int64_t> dims;
    for (auto dim : absl::StrSplit(tile_shape, ',')) {
      dims.push_back(0);
      if (!absl::SimpleAtoi(dim, &dims.back()) || dims.back() <= 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid tile shape: ", tile_shape));
      }
    }
    tile_config = TilingConfig{std::move(dims), 0};
  }

  // Set up the XLA StreamExecutor runtime used to execute the TensorIR
  // kernels.
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                        PlatformUtil::GetDefaultPlatform());
  ABSL_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        platform->ExecutorForDevice(0));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                        executor->CreateStream());

  // Single or multiple fusion benchmark.
  return single ? BenchmarkCompiledHLO(input_file, options, executor,
                                       stream.get(), repeats, autotune,
                                       tile_config)
                : BenchmarkFullHLO(input_file, options, executor, stream.get(),
                                   repeats, autotune, output_path);
}

}  // namespace
}  // namespace xla::gpu::tensor_ir

class VLogStderrSink final : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    if (entry.verbosity() >= 1) {
      llvm::errs() << entry.text_message_with_prefix_and_newline();
    }
  }
};

int main(int argc, char** argv) {
  std::string target = "sm_100f";
  int64_t repeats = 20;
  int64_t autotune = 0;
  bool single = false;
  std::string tile_shape = "";
  std::string output_path = "";

  std::vector<tsl::Flag> flags = {
      tsl::Flag("target", &target, "CudaTile target SM."),
      tsl::Flag("repeats", &repeats, "Number of benchmark repetitions."),
      tsl::Flag("autotune", &autotune, "Autotune (number of tilings)."),
      tsl::Flag("single", &single, "Benchmark a single fusion."),
      tsl::Flag("tile_shape", &tile_shape, "Tile shape (single fusion only)."),
      tsl::Flag("output_path", &output_path, "Path to the results directory."),
  };

  const std::string usage = tsl::Flags::Usage(argv[0], flags);
  tsl::port::InitMain(argv[0], &argc, &argv);
  LOG_IF(QFATAL, !tsl::Flags::Parse(&argc, argv, flags)) << usage;

  VLogStderrSink sink;
  absl::AddLogSink(&sink);
  absl::SetStderrThreshold(absl::LogSeverity::kError);
  absl::InitializeLog();

  for (int i = 1; i < argc; ++i) {
    absl::Status status = xla::gpu::tensor_ir::RealMain(
        argv[i], target, repeats, autotune, single, tile_shape, output_path);
    LOG_IF(QFATAL, !status.ok()) << status;
  }
  return 0;
}
