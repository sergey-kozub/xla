#ifndef TENSOR_IR_ANALYSIS_TILING_ENUMERATE_H_
#define TENSOR_IR_ANALYSIS_TILING_ENUMERATE_H_

#include "xla/backends/gpu/codegen/tensor_ir/temp/Evaluate.h"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

/**
 * @brief Enumerate tilings for the given graph (bounded search).
 * @param graph The TensorIR graph to analyze.
 * @param maxResults The maximum number of results to return.
 */
FailureOr<SmallVector<TilingConfig>>
enumerateTilings(GraphOp graph, int64_t maxResults = 1000);

} // namespace tiling_analysis
} // namespace mlir::nv_tensor_ir

#endif // TENSOR_IR_ANALYSIS_TILING_ENUMERATE_H_
