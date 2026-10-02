#ifndef TENSOR_IR_ANALYSIS_TILING_EVALUATE_H_
#define TENSOR_IR_ANALYSIS_TILING_EVALUATE_H_

#include "tensor_ir/Dialect/TensorIR.h"
#include "xla/backends/gpu/codegen/tensor_ir/temp/Arch.h"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

/**
 * @brief Tiling configuration.
 */
struct TilingConfig {
  SmallVector<int64_t> tileShape;
  int64_t reductionTileSize;

  bool validate() const;
  std::string toString() const;
};

/**
 * @brief Tiling evaluation result.
 */
struct TilingEvaluation {
  // Number of execution blocks.
  int64_t blockCount = 1;

  // Estimate storage size for the tiles in the computation graph.
  // The heuristic doesn't run the liveness analysis, so it may underestimate
  // the actual usage, as input tiles and intermediate tiles may be alive at
  // the same time.
  int64_t tileStorageBytes = 0;

  // Number of memory accesses (loads/stores).
  int64_t memoryAccessCount = 0;

  // Estimate number of cache lines accessed.
  int64_t memoryAccessCacheLines = 0;

  // Number of MMA operations.
  int64_t mmaOperationCount = 0;

  // Error status; if not successful then other fields are undefined.
  enum class EvalError {
    Success,
    MissingOrInvalidLayout,
    InvalidTilingConfig,
    MissingTensorElementSize,
    ConcatDimensionIsTiled,
  } errorStatus = EvalError::Success;

  std::string toString() const;
};

/**
 * @brief Tiling evaluator implementation.
 */
class TilingEvaluator {
 public:
  TilingEvaluator(const ArchInfo& archInfo) : archInfo(archInfo) {}

  /**
   * @brief Evaluate the tiling configuration for a given layout.
   * @param graph The TensorIR graph to evaluate.
   * @param tilingConfig The tiling configuration.
   */
  TilingEvaluation evaluate(GraphOp graph,
                            const TilingConfig& tilingConfig) const;

  /**
   * @brief Estimate the occupancy based on the evaluation result.
   * @param evaluation The evaluation result.
   * @param warpCount The number of warps per block.
   */
  int estimateOccupancy(const TilingEvaluation& evaluation,
                        int warpCount) const;

 private:
  const ArchInfo& archInfo;
};

// Helpers for getting input/output layouts from the graph.
LayoutSourceAttrInterface getInputLayout(GraphOp graph);
LayoutSourceAttrInterface getOutputLayout(GraphOp graph);

}  // namespace tiling_analysis
}  // namespace mlir::nv_tensor_ir

#endif  // TENSOR_IR_ANALYSIS_TILING_EVALUATE_H_
