#include "xla/backends/gpu/codegen/tensor_ir/temp/Enumerate.h"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {
namespace {

// Minimum tile size (1 warp, 1 value per thread).
constexpr int kMinTileSizeBits = 5;  // 2^5 = 32
// Maximum tile size (8 warps, 64 values per thread).
constexpr int kMaxTileSizeBits = 14;  // 2^14 = 16384
// Maximum tile size for dynamic dimensions.
constexpr int kMaxDynamicTileSizeBits = 6;  // 2^6 = 64
// Minimum reduction tile size.
constexpr int kMinReductionTileSizeBits = 5;  // 2^5 = 32

// Tile size limits for matmul dimensions.
constexpr int kMinLhsTileSizeBits = 4;          // 2^4 = 16
constexpr int kMaxLhsTileSizeBits = 7;          // 2^7 = 128
constexpr int kMinRhsTileSizeBits = 3;          // 2^3 = 8
constexpr int kMaxRhsTileSizeBits = 8;          // 2^8 = 256
constexpr int kMinContractionTileSizeBits = 4;  // 2^4 = 16
constexpr int kMaxContractionTileSizeBits = 7;  // 2^7 = 128

/**
 * @brief Information about a dimension in the main iteration space.
 */
struct DimensionInfo {
  // Dimension number, -1 for reduction dimensions.
  int index;

  // Number of tensors where this dimension can be vectorized (stride=1).
  int vectorizableCount = 0;

  // Number of tensors where this dimension is broadcasted (stride=0).
  int broadcastedCount = 0;

  // Marks dimensions that cannot be tiled.
  bool nonTilable = false;

  // Marks dimensions that propagate to a matmul result.
  bool usedInMatmulM = false;
  bool usedInMatmulN = false;
};

/**
 * @brief Collect dimension information from a layout.
 */
void collectDimensionInfo(LayoutSourceAttrInterface layout,
                          ArrayRef<DimensionInfo*> dimensionData) {
  return llvm::TypeSwitch<LayoutSourceAttrInterface>(layout)
      .Case<TensorSourceAttr>([&](TensorSourceAttr tensor) {
        // Process tensor layout.
        tcutegen::Layout cuteLayout = tensor.getCuteLayout();
        for (auto [idx, stride] :
             llvm::enumerate(cuteLayout.stride().getValues())) {
          if (stride.isStatic()) {
            if (stride.as_int() == 1) {
              ++dimensionData[idx]->vectorizableCount;
            } else if (stride.as_int() == 0) {
              ++dimensionData[idx]->broadcastedCount;
            }
          }
        }
      })
      .Case<CompositeSourceAttr>([&](CompositeSourceAttr composite) {
        // Process composite layout.
        for (LayoutSourceAttrInterface src : composite.getSources()) {
          collectDimensionInfo(src, dimensionData);
        }
      })
      .Case<ConcatSourceAttr>([&](ConcatSourceAttr concat) {
        // Process concat layout.
        dimensionData[concat.getDimension()]->nonTilable = true;
        for (LayoutSourceAttrInterface src : concat.getSources()) {
          collectDimensionInfo(src, dimensionData);
        }
      })
      .Case<ReductionSourceAttr>([&](ReductionSourceAttr reduction) {
        // Process reduction layout.
        DimensionInfo reductionDimension{.index = -1};
        SmallVector<DimensionInfo*> newDimensionData(dimensionData);
        newDimensionData.resize(
            dimensionData.size() + reduction.getReductionShape().size(),
            &reductionDimension);
        collectDimensionInfo(reduction.getSource(), newDimensionData);
      })
      .Case<MatmulSourceAttr>([&](MatmulSourceAttr matmul) {
        // Process matmul layout.
        DimensionInfo reductionDimension{.index = -1};
        size_t rank = dimensionData.size();

        tcutegen::Layout cuteLayout = matmul.getCuteLayout();
        for (auto [idx, stride] :
             llvm::enumerate(cuteLayout.stride().getValues())) {
          if (stride.isStatic()) {
            int64_t pos = stride.as_int();
            if (pos < matmul.getM() * matmul.getN() * matmul.getK()) {
              if (pos >= matmul.getN() * matmul.getK()) {
                dimensionData[idx]->usedInMatmulM = true;
              } else if (pos >= matmul.getK()) {
                dimensionData[idx]->usedInMatmulN = true;
              }
            }
          }
        }

        // Process LHS input.
        if (auto lhsDimensionMap = matmul.getLhsDimensionMap();
            succeeded(lhsDimensionMap)) {
          SmallVector<DimensionInfo*> lhsDimensionData;
          for (size_t idx : *lhsDimensionMap) {
            lhsDimensionData.push_back(idx < rank ? dimensionData[idx]
                                                  : &reductionDimension);
          }
          collectDimensionInfo(matmul.getLhs(), lhsDimensionData);
        }

        // Process RHS input.
        if (auto rhsDimensionMap = matmul.getRhsDimensionMap();
            succeeded(rhsDimensionMap)) {
          SmallVector<DimensionInfo*> rhsDimensionData;
          for (size_t idx : *rhsDimensionMap) {
            rhsDimensionData.push_back(idx < rank ? dimensionData[idx]
                                                  : &reductionDimension);
          }
          collectDimensionInfo(matmul.getRhs(), rhsDimensionData);
        }
      })
      .Default([](LayoutSourceAttrInterface) {});
}

/**
 * @brief Get the maximum reduction size from the graph.
 */
int64_t getMaxReductionSize(LayoutSourceAttrInterface layout) {
  return llvm::TypeSwitch<LayoutSourceAttrInterface, int64_t>(layout)
      .Case<TensorSourceAttr>([](TensorSourceAttr) { return 1; })
      .Case<CompositeSourceAttr>([&](CompositeSourceAttr composite) {
        return *llvm::max_element(
            llvm::map_range(composite.getSources(), getMaxReductionSize));
      })
      .Case<ConcatSourceAttr>([&](ConcatSourceAttr concat) {
        return *llvm::max_element(
            llvm::map_range(concat.getSources(), getMaxReductionSize));
      })
      .Case<ReductionSourceAttr>([&](ReductionSourceAttr reduction) {
        return std::max(reduction.getReductionSize(),
                        getMaxReductionSize(reduction.getSource()));
      })
      .Case<MatmulSourceAttr>([&](MatmulSourceAttr matmul) {
        return std::max(getMaxReductionSize(matmul.getLhs()),
                        getMaxReductionSize(matmul.getRhs()));
      })
      .Default([](LayoutSourceAttrInterface) { return 0; });
}

/**
 * @brief Get the dimension information for the main iteration space.
 * @param input The layout of the inputs propagated to the output.
 * @param output The layout of the output.
 */
FailureOr<SmallVector<DimensionInfo>> getDimensionInfo(
    LayoutSourceAttrInterface input, LayoutSourceAttrInterface output) {
  auto inputShape = input.getShape();
  if (!output || inputShape != output.getShape()) {
    return failure();
  }

  SmallVector<DimensionInfo> dimensionInfo;
  SmallVector<DimensionInfo*> dimensionInfoPtr;
  dimensionInfo.reserve(inputShape.size());
  for (int idx : llvm::seq(inputShape.size())) {
    dimensionInfo.push_back({.index = idx});
    dimensionInfoPtr.push_back(&dimensionInfo.back());
  }

  collectDimensionInfo(input, dimensionInfoPtr);
  collectDimensionInfo(output, dimensionInfoPtr);
  return dimensionInfo;
}

/**
 * @brief Generate all combinations of bits assigned into slots.
 * @param maxBits The maximum number of bits for each dimension.
 * @param totalBits The total number of bits to assign.
 * @param maxResults The maximum number of results to return.
 */
SmallVector<SmallVector<int64_t>> generateCombinations(ArrayRef<int> maxBits,
                                                       int totalBits,
                                                       int64_t maxResults) {
  // Calculate suffix sum for early stopping.
  SmallVector<int> suffixSum(maxBits.size(), 0);
  for (int i = static_cast<int>(maxBits.size()) - 1; i > 0; --i) {
    suffixSum[i - 1] = suffixSum[i] + maxBits[i];
  }

  // Generate combinations recursively.
  SmallVector<SmallVector<int64_t>> result;
  SmallVector<int64_t> current(maxBits.size(), 0);
  std::function<void(size_t, int)> generate = [&](size_t idx, int bits) {
    if (result.size() == static_cast<size_t>(maxResults)) {
      return;
    }
    // All bits are assigned, add the current combination.
    if (bits == 0) {
      result.push_back(current);
      return;
    }
    // Try all possible assignments for the current dimension.
    // Start from the higher value, as the dimensions are sorted.
    for (int i = std::min(bits, maxBits[idx]);
         i >= std::max(bits - suffixSum[idx], 0); --i) {
      current[idx] = i;
      generate(idx + 1, bits - i);
    }
    current[idx] = 0;
  };
  generate(0, totalBits);
  return result;
}

/**
 * @brief Generate all combinations of bits in the given range.
 * @param maxBits The maximum number of bits for each dimension.
 * @param lowerBound The minimum total number of bits.
 * @param upperBound The maximum total number of bits.
 * @param maxResults The maximum number of results to return.
 */
SmallVector<SmallVector<int64_t>> generateCombinations(ArrayRef<int> maxBits,
                                                       int lowerBound,
                                                       int upperBound,
                                                       int64_t maxResults) {
  // If there are no dimensions, adjust the upper bound.
  if (maxBits.empty()) {
    upperBound = 0;
  }

  // Generate the combinations with each bit count in the range.
  SmallVector<SmallVector<int64_t>> result;
  for (int bits = upperBound; bits >= lowerBound; --bits) {
    int64_t limit = llvm::divideCeil(maxResults, bits - lowerBound + 1);
    SmallVector<SmallVector<int64_t>> combinations =
        generateCombinations(maxBits, bits, limit);
    maxResults -= combinations.size();
    result.append(combinations);
  }

  // Fall back to the unit tile size if the result is empty.
  if (result.empty()) {
    result.push_back(SmallVector<int64_t>(maxBits.size(), 0));
  }
  return result;
}

/**
 * @brief Search for tilings with the given constraints.
 * @param dimensionInfo The dimension information for the layout.
 * @param maxTileSizeBits The maximum tile size for each dimension (bits).
 * @param reductionTileSizeBits The reduction tile size (bits).
 * @param maxResults The maximum number of results to return.
 * @param hasMatmul Set if a matmul is present in the graph.
 */
SmallVector<TilingConfig> searchTilings(ArrayRef<DimensionInfo> dimensionInfo,
                                        ArrayRef<int> maxTileSizeBits,
                                        int reductionTileSizeBits,
                                        int64_t maxResults, bool hasMatmul) {
  if (hasMatmul) {
    // Filter dimensions based on a predicate.
    auto filterDimensions = [&](auto predicate) {
      SmallVector<size_t> dimIndex;
      SmallVector<int> dimLimit;
      for (const auto& [dimInfo, maxBits] :
           llvm::zip_equal(dimensionInfo, maxTileSizeBits)) {
        if (predicate(dimInfo)) {
          dimIndex.push_back(dimInfo.index);
          dimLimit.push_back(maxBits);
        }
      }
      return std::make_pair(dimIndex, dimLimit);
    };

    // Split the dimensions by type.
    auto [lhsIndex, lhsLimit] = filterDimensions(
        [&](const DimensionInfo& dimInfo) { return dimInfo.usedInMatmulM; });
    auto [rhsIndex, rhsLimit] = filterDimensions(
        [&](const DimensionInfo& dimInfo) { return dimInfo.usedInMatmulN; });
    auto [otherIndex, otherLimit] =
        filterDimensions([&](const DimensionInfo& dimInfo) {
          return !dimInfo.usedInMatmulM && !dimInfo.usedInMatmulN;
        });

    // Enumerate M/N dimension tilings.
    int64_t maxSideResults = std::sqrt(maxResults);
    auto lhsCombinations = generateCombinations(
        lhsLimit, kMinLhsTileSizeBits,
        std::min(kMaxLhsTileSizeBits, kMaxTileSizeBits - reductionTileSizeBits),
        maxSideResults);
    auto rhsCombinations = generateCombinations(
        rhsLimit, kMinRhsTileSizeBits,
        std::min(kMaxRhsTileSizeBits, kMaxTileSizeBits - reductionTileSizeBits),
        maxSideResults);

    // Tile size accumulator.
    SmallVector<int64_t> tileSize(dimensionInfo.size());
    auto updateTileSize = [&](ArrayRef<int64_t> combination,
                              ArrayRef<size_t> dimIndex) {
      int64_t bits = 0;
      for (size_t idx : llvm::seq(combination.size())) {
        tileSize[dimIndex[idx]] = 1 << combination[idx];
        bits += combination[idx];
      }
      return bits;
    };

    // Enumerate all possible M/N combinations.
    SmallVector<TilingConfig> result;
    int64_t rest = lhsCombinations.size() * rhsCombinations.size();
    for (auto& lhsCombination : lhsCombinations) {
      int64_t lhsBits = updateTileSize(lhsCombination, lhsIndex);
      for (auto& rhsCombination : rhsCombinations) {
        int64_t rhsBits = updateTileSize(rhsCombination, rhsIndex);
        int64_t otherBits = kMaxTileSizeBits - reductionTileSizeBits -
                            std::max(lhsBits, rhsBits);
        auto combinations =
            generateCombinations(otherLimit, 0, otherBits, maxResults / rest);
        for (auto& combination : combinations) {
          updateTileSize(combination, otherIndex);
          result.push_back(TilingConfig{tileSize, 1 << reductionTileSizeBits});
        }
        maxResults -= combinations.size();
        --rest;
      }
    }
    if (!result.empty()) {
      return result;
    }
  }

  // Distribute the tile size bits in the valid range.
  SmallVector<SmallVector<int64_t>> combinations = generateCombinations(
      maxTileSizeBits, kMinTileSizeBits, kMaxTileSizeBits, maxResults);

  // Build the tilings from the combinations.
  SmallVector<TilingConfig> result;
  for (auto& combination : combinations) {
    SmallVector<int64_t> tileSize(combination.size(), 0);
    for (size_t idx : llvm::seq(combination.size())) {
      tileSize[dimensionInfo[idx].index] = 1 << combination[idx];
    }
    result.push_back(
        TilingConfig{std::move(tileSize), 1 << reductionTileSizeBits});
  }
  return result;
}

}  // namespace

FailureOr<SmallVector<TilingConfig>> enumerateTilings(GraphOp graph,
                                                      int64_t maxResults) {
  // Get the dimension information for the layout.
  auto inputLayout = getInputLayout(graph);
  auto layoutShape = inputLayout.getShape();
  auto outputLayout = getOutputLayout(graph).reshape(layoutShape);
  auto dimensionInfo = getDimensionInfo(inputLayout, outputLayout);
  if (failed(dimensionInfo)) {
    return failure();
  }

  // Sort the dimensions, bring important dimensions to the front.
  auto compare = [](const DimensionInfo& a, const DimensionInfo& b) {
    if (a.vectorizableCount != b.vectorizableCount) {
      return a.vectorizableCount > b.vectorizableCount;
    }
    if (a.broadcastedCount != b.broadcastedCount) {
      return a.broadcastedCount > b.broadcastedCount;
    }
    return a.index < b.index;
  };
  llvm::sort(*dimensionInfo, compare);

  // Build the maximum tile size (bits) for each dimension.
  SmallVector<int> maxTileSizeBits;
  for (const auto& dimInfo : *dimensionInfo) {
    int64_t size = layoutShape[dimInfo.index];
    int maxBits = size != ShapedType::kDynamic ? llvm::Log2_64_Ceil(size)
                                               : kMaxDynamicTileSizeBits;
    maxTileSizeBits.push_back(!dimInfo.nonTilable ? maxBits : 0);
  }

  // Check if matmuls or reductions are present in the graph.
  // If reductions are present, set the upper bound at the reduction size.
  bool hasMatmul = !graph.getOps<MatmulOp>().empty();
  bool hasReduction =
      !graph.getOps<ReduceOp>().empty() || !graph.getOps<ReduceUDOp>().empty();

  int maxReductionBits =
      hasReduction ? llvm::Log2_64_Ceil(getMaxReductionSize(inputLayout)) : 0;
  int minReductionBits = std::min(kMinReductionTileSizeBits, maxReductionBits);
  maxReductionBits = std::min(kMaxTileSizeBits, maxReductionBits);

  // Set lower/upper bound for the tile size (bits) for matmuls/reductions.
  int lowerBound = 0, upperBound = 0;
  if (hasMatmul && hasReduction) {
    // Both matmuls and reductions are present, extend the normal matmul range
    // to the reduction range.
    lowerBound = std::min(kMinContractionTileSizeBits, minReductionBits);
    upperBound = maxReductionBits;
  } else if (hasMatmul) {
    // Only matmuls are present, use the normal matmul range.
    lowerBound = kMinContractionTileSizeBits;
    upperBound = kMaxContractionTileSizeBits;
  } else if (hasReduction) {
    // Only reductions are present, use the normal reduction range.
    lowerBound = minReductionBits;
    upperBound = maxReductionBits;
  }

  // Try all possible reduction tile sizes starting from the higher value,
  // i.e. in the order of increasing the number of bit combinations.
  SmallVector<TilingConfig> result;
  for (int bits = upperBound; bits >= lowerBound; --bits) {
    int limit = llvm::divideCeil(maxResults, bits - lowerBound + 1);
    SmallVector<TilingConfig> tilings =
        searchTilings(*dimensionInfo, maxTileSizeBits, bits, limit, hasMatmul);
    maxResults -= tilings.size();
    result.append(tilings);
  }
  return result;
}

}  // namespace tiling_analysis
}  // namespace mlir::nv_tensor_ir
