#include "xla/backends/gpu/codegen/tensor_ir/temp/Evaluate.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tiling-analysis-evaluate"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {
namespace {

TilingEvaluation evaluateImpl(LayoutSourceAttrInterface layout,
                              const TilingConfig& tilingConfig,
                              ArrayRef<size_t> tensorElementSizes,
                              const ArchInfo& archInfo);

// Size value used for dynamic dimensions.
constexpr int64_t kDynamicSizeValue = 1024;

// Element size used for reduction accumulators (bytes).
constexpr int64_t kReductionAccumulatorElementSize = 4;

/// Calculate the reduction tile shape by distributing the reduction tile size
/// evenly between the contracting dimensions.
SmallVector<int64_t> calculateReductionTileShape(
    ArrayRef<int64_t> reductionShape, int64_t reductionTileSize) {
  SmallVector<int64_t> tileShape;
  int rest = llvm::Log2_64(reductionTileSize);
  int dist = reductionShape.size();

  for (int64_t size : reductionShape) {
    int bits = std::min(llvm::Log2_64_Ceil(uint64_t(size)),
                        llvm::divideCeil(rest, dist--));
    rest -= bits;
    tileShape.push_back(1ull << bits);
  }
  return tileShape;
}

/**
 * @brief Tensor layout analysis.
 */
TilingEvaluation evaluate(TensorSourceAttr layout,
                          const TilingConfig& tilingConfig,
                          ArrayRef<size_t> tensorElementSizes) {
  // Negative tensor IDs denote constant tensors.
  if (layout.getTensorId() < 0) {
    return TilingEvaluation{};
  }

  // If there's no size information for the tensor, return an error.
  if (size_t(layout.getTensorId()) >= tensorElementSizes.size()) {
    return TilingEvaluation{
        .errorStatus = TilingEvaluation::EvalError::MissingTensorElementSize,
    };
  }

  // Calculate the tile size in bytes.
  int elementSizeInBytes = tensorElementSizes[layout.getTensorId()];
  int64_t tileSizeInBytes =
      llvm::product_of(tilingConfig.tileShape) * elementSizeInBytes;

  // If the base pointer is offset and doesn't lie on a cache line boundary,
  // an additional cache line access may be needed.
  int elementsPerCacheLine = kCacheLineSize / elementSizeInBytes;
  bool isUnalignedPointer = layout.getOffset() % elementsPerCacheLine != 0;

  // Calculate the product of cache line accesses for each dimension.
  tcutegen::Layout cuteLayout = layout.getCuteLayout();
  int64_t cacheLineAccesses = 1;
  for (auto [idx, tileSize] : llvm::enumerate(tilingConfig.tileShape)) {
    int64_t cacheLinesInDim = tileSize;

    // If stride is not static, assume the worst case (distinct cache lines).
    const auto& cgStride = cuteLayout.stride().get(idx);
    if (cgStride.isStatic() && tileSize > 1) {
      int64_t stride = cgStride.as_int();
      if (stride == 0) {
        // Broadcasted dimensions are not accessed.
        continue;
      }
      if (stride < elementsPerCacheLine) {
        // If the reads don't align on cache line boundaries, an additional
        // cache line access may be needed.
        const auto& cgShape = cuteLayout.shape().get(idx);
        int64_t size = cgShape.isStatic() ? cgShape.as_int() : INT64_MAX;
        int64_t readSize = std::min(size, tileSize) * stride;
        bool isUnalignedDimension = readSize % elementsPerCacheLine != 0 &&
                                    elementsPerCacheLine % readSize != 0;
        cacheLinesInDim =
            llvm::divideCeil(readSize - stride + 1, elementsPerCacheLine) +
            (isUnalignedPointer || isUnalignedDimension);
      }
    }
    cacheLineAccesses *= cacheLinesInDim;
  }

  return TilingEvaluation{
      .tileStorageBytes = tileSizeInBytes,
      .memoryAccessCount = 1,
      .memoryAccessCacheLines = cacheLineAccesses,
  };
}

/**
 * @brief Composite layout analysis.
 */
TilingEvaluation evaluate(CompositeSourceAttr layout,
                          const TilingConfig& tilingConfig,
                          ArrayRef<size_t> tensorElementSizes,
                          const ArchInfo& archInfo) {
  // Accumulate the results from the child layouts.
  TilingEvaluation result;
  for (LayoutSourceAttrInterface src : layout.getSources()) {
    TilingEvaluation item =
        evaluateImpl(src, tilingConfig, tensorElementSizes, archInfo);
    if (item.errorStatus != TilingEvaluation::EvalError::Success) {
      return item;
    }
    result.tileStorageBytes += item.tileStorageBytes;
    result.memoryAccessCount += item.memoryAccessCount;
    result.memoryAccessCacheLines += item.memoryAccessCacheLines;
    result.mmaOperationCount += item.mmaOperationCount;
  }
  return result;
}

/**
 * @brief Concat layout analysis.
 */
TilingEvaluation evaluate(ConcatSourceAttr layout,
                          const TilingConfig& tilingConfig,
                          ArrayRef<size_t> tensorElementSizes,
                          const ArchInfo& archInfo) {
  // Verify the tiling configuration (concat dimension must have unit size).
  if (tilingConfig.tileShape[layout.getDimension()] != 1) {
    return TilingEvaluation{
        .errorStatus = TilingEvaluation::EvalError::ConcatDimensionIsTiled,
    };
  }

  // Accumulate the results from the child layouts.
  TilingEvaluation result;
  int64_t totalWeight = 0;

  for (LayoutSourceAttrInterface src : layout.getSources()) {
    TilingEvaluation item =
        evaluateImpl(src, tilingConfig, tensorElementSizes, archInfo);
    if (item.errorStatus != TilingEvaluation::EvalError::Success) {
      return item;
    }

    // Calculate the maximum tile storage in the branches.
    result.tileStorageBytes =
        std::max(result.tileStorageBytes, item.tileStorageBytes);
    result.mmaOperationCount =
        std::max(result.mmaOperationCount, item.mmaOperationCount);

    // Calculate the weighted values across the branches.
    int64_t weight = src.getShape()[layout.getDimension()];
    if (weight == ShapedType::kDynamic) {
      weight = kDynamicSizeValue;
    }
    result.memoryAccessCount += weight * item.memoryAccessCount;
    result.memoryAccessCacheLines += weight * item.memoryAccessCacheLines;
    totalWeight += weight;
  }

  // Normalize the weighted values.
  result.memoryAccessCount /= totalWeight;
  result.memoryAccessCacheLines /= totalWeight;
  return result;
}

/**
 * @brief Reduction layout analysis.
 */
TilingEvaluation evaluate(ReductionSourceAttr layout,
                          const TilingConfig& tilingConfig,
                          ArrayRef<size_t> tensorElementSizes,
                          const ArchInfo& archInfo) {
  // Verify the tiling configuration (reduction tile size).
  if (tilingConfig.reductionTileSize <= 0) {
    return TilingEvaluation{
        .errorStatus = TilingEvaluation::EvalError::InvalidTilingConfig,
    };
  }

  // Calculate the number of iterations.
  SmallVector<int64_t> reductionTileShape = calculateReductionTileShape(
      layout.getReductionShape(), tilingConfig.reductionTileSize);
  int64_t repeats = 1;
  for (auto [size, tileSize] :
       llvm::zip_equal(layout.getReductionShape(), reductionTileShape)) {
    repeats *= llvm::divideCeil(size, tileSize);
  }

  // Build the new tiling configuration for the child layout.
  SmallVector<int64_t> tilingShape;
  auto cuteLayout = layout.getCuteLayout();
  auto strideValues = cuteLayout.stride().getValues().drop_back();
  for (auto [idx, stride] : llvm::enumerate(strideValues)) {
    if (stride.isStatic() && stride.as_int() > 0) {
      tilingShape.push_back(tilingConfig.tileShape[idx]);
    }
  }
  tilingShape.append(reductionTileShape);
  TilingConfig config{tilingShape, tilingConfig.reductionTileSize};

  // Accumulator is only needed if a loop is present.
  int64_t accumulatorSize = repeats > 1 ? llvm::product_of(config.tileShape) *
                                              kReductionAccumulatorElementSize
                                        : 0;

  // Evaluate the child layout and adjust results.
  TilingEvaluation result =
      evaluateImpl(layout.getSource(), config, tensorElementSizes, archInfo);

  result.tileStorageBytes += accumulatorSize;
  result.memoryAccessCount *= repeats;
  result.memoryAccessCacheLines *= repeats;
  return result;
}

// Check if MMA v5 (e.g. Blackwell) can be used.
bool isValidMMAv5(int64_t tileM, int64_t tileN, int64_t tileK,
                  const ArchInfo& archInfo) {
  bool allowMMA = (archInfo.tensorCoreVersions & kMMAv5) != 0 &&
                  (tileM == 64 || tileM == 128) &&
                  (tileN % 8 == 0 && tileN <= 256) && (tileK % 16 == 0);

  // Assume `kind::f16` for calculating the shared memory size.
  // Only RHS must be in the shared memory.
  constexpr int64_t kInputElementSize = 2;
  int64_t sharedMemorySize = tileN * tileK * kInputElementSize;

  return allowMMA && sharedMemorySize <= archInfo.maxSharedMemoryPerSM * 1024;
}

// Check if MMA v3 (e.g. Hopper) can be used.
bool isValidMMAv3(int64_t tileM, int64_t tileN, int64_t tileK,
                  const ArchInfo& archInfo) {
  bool allowMMA = (archInfo.tensorCoreVersions & kMMAv3) != 0 && tileM == 64 &&
                  (tileN % 8 == 0 && tileN <= 256) && (tileK % 16 == 0);

  // Assume `kind::f16` for calculating the shared memory size.
  // Only RHS must be in the shared memory.
  constexpr int64_t kInputElementSize = 2;
  int64_t sharedMemorySize = tileN * tileK * kInputElementSize;

  return allowMMA && sharedMemorySize <= archInfo.maxSharedMemoryPerSM * 1024;
}

// Check if MMA v2 (e.g. Ampere) can be used.
bool isValidMMAv2(int64_t tileM, int64_t tileN, int64_t tileK,
                  const ArchInfo& archInfo) {
  // Assume `.m16n8k16` MMA tile shape.
  return (archInfo.tensorCoreVersions & kMMAv2) != 0 && tileM % 16 == 0 &&
         tileN % 8 == 0 && tileK % 16 == 0;
}

/**
 * @brief Matmul layout analysis.
 */
TilingEvaluation evaluate(MatmulSourceAttr layout,
                          const TilingConfig& tilingConfig,
                          ArrayRef<size_t> tensorElementSizes,
                          const ArchInfo& archInfo) {
  // Verify the tiling configuration (reduction tile size).
  if (tilingConfig.reductionTileSize <= 0) {
    return TilingEvaluation{
        .errorStatus = TilingEvaluation::EvalError::InvalidTilingConfig,
    };
  }

  // Calculate the number of iterations.
  SmallVector<int64_t> reductionTileShape = calculateReductionTileShape(
      layout.getContractingShape(), tilingConfig.reductionTileSize);
  int64_t repeats = 1;
  for (auto [size, tileSize] :
       llvm::zip_equal(layout.getContractingShape(), reductionTileShape)) {
    repeats *= llvm::divideCeil(size, tileSize);
  }

  // Get the dimension map for the LHS and RHS.
  size_t rank = tilingConfig.tileShape.size();
  auto lhsDimensionMap = layout.getLhsDimensionMap();
  auto rhsDimensionMap = layout.getRhsDimensionMap();
  if (failed(lhsDimensionMap) || failed(rhsDimensionMap)) {
    return TilingEvaluation{
        .errorStatus = TilingEvaluation::EvalError::MissingOrInvalidLayout,
    };
  }

  // Evaluate LHS.
  SmallVector<int64_t> lhsTileShape;
  for (size_t idx : *lhsDimensionMap) {
    lhsTileShape.push_back(idx < rank ? tilingConfig.tileShape[idx]
                                      : reductionTileShape[idx - rank]);
  }
  TilingConfig lhsConfig{lhsTileShape, tilingConfig.reductionTileSize};
  TilingEvaluation lhsEval =
      evaluateImpl(layout.getLhs(), lhsConfig, tensorElementSizes, archInfo);

  // Evaluate RHS.
  SmallVector<int64_t> rhsTileShape;
  for (size_t idx : *rhsDimensionMap) {
    rhsTileShape.push_back(idx < rank ? tilingConfig.tileShape[idx]
                                      : reductionTileShape[idx - rank]);
  }
  TilingConfig rhsConfig{rhsTileShape, tilingConfig.reductionTileSize};
  TilingEvaluation rhsEval =
      evaluateImpl(layout.getRhs(), rhsConfig, tensorElementSizes, archInfo);

  // Calculate tile B/M/N/K sizes.
  int64_t tileB = 1, tileM = 1, tileN = 1,
          tileK = tilingConfig.reductionTileSize;
  auto cuteLayout = layout.getCuteLayout();
  for (auto [idx, stride] : llvm::enumerate(cuteLayout.stride().getValues())) {
    int64_t pos = stride.as_int();
    if (pos >= layout.getM() * layout.getN() * layout.getK()) {
      tileB *= tilingConfig.tileShape[idx];
    } else if (pos >= layout.getN() * layout.getK()) {
      tileM *= tilingConfig.tileShape[idx];
    } else if (pos >= layout.getK()) {
      tileN *= tilingConfig.tileShape[idx];
    }
  }

  // Count the number of MMA operations.
  int mmaOperationCount = 0;
  if (isValidMMAv5(tileM, tileN, tileK, archInfo)) {
    mmaOperationCount = tileB * tileK / 16;
  } else if (isValidMMAv3(tileM, tileN, tileK, archInfo)) {
    mmaOperationCount = tileB * tileK / 16;
  } else if (isValidMMAv2(tileM, tileN, tileK, archInfo)) {
    mmaOperationCount = tileB * (tileM / 16) * (tileN / 8) * (tileK / 16);
  }

  // Assume the accumulator is kept in registers.
  int64_t accumulatorSize = llvm::product_of(tilingConfig.tileShape) *
                            kReductionAccumulatorElementSize;

  // Merge the results.
  return TilingEvaluation{
      .tileStorageBytes =
          lhsEval.tileStorageBytes + rhsEval.tileStorageBytes + accumulatorSize,
      .memoryAccessCount =
          repeats * (lhsEval.memoryAccessCount + rhsEval.memoryAccessCount),
      .memoryAccessCacheLines = repeats * (lhsEval.memoryAccessCacheLines +
                                           rhsEval.memoryAccessCacheLines),
      .mmaOperationCount = mmaOperationCount,
  };
}

/**
 * @brief Dispatch to the appropriate evaluation function.
 */
TilingEvaluation evaluateImpl(LayoutSourceAttrInterface layout,
                              const TilingConfig& tilingConfig,
                              ArrayRef<size_t> tensorElementSizes,
                              const ArchInfo& archInfo) {
  return llvm::TypeSwitch<LayoutSourceAttrInterface, TilingEvaluation>(layout)
      .Case<TensorSourceAttr>([&](TensorSourceAttr tensor) {
        return evaluate(tensor, tilingConfig, tensorElementSizes);
      })
      .Case<CompositeSourceAttr>([&](CompositeSourceAttr composite) {
        return evaluate(composite, tilingConfig, tensorElementSizes, archInfo);
      })
      .Case<ConcatSourceAttr>([&](ConcatSourceAttr concat) {
        return evaluate(concat, tilingConfig, tensorElementSizes, archInfo);
      })
      .Case<ReductionSourceAttr>([&](ReductionSourceAttr reduction) {
        return evaluate(reduction, tilingConfig, tensorElementSizes, archInfo);
      })
      .Case<MatmulSourceAttr>([&](MatmulSourceAttr matmul) {
        return evaluate(matmul, tilingConfig, tensorElementSizes, archInfo);
      })
      .Default([](LayoutSourceAttrInterface) { return TilingEvaluation{}; });
}

}  // namespace

bool TilingConfig::validate() const {
  // Reduction size must be a power of two, if specified.
  if (reductionTileSize > 0 && !llvm::isPowerOf2_64(reductionTileSize)) {
    return false;
  }
  // All tile sizes must be a power of two.
  return llvm::all_of(tileShape, [](int64_t tileSize) {
    return tileSize > 0 && llvm::isPowerOf2_64(tileSize);
  });
}

std::string TilingConfig::toString() const {
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << "[";
  llvm::interleave(tileShape, ss, ",");
  ss << "]";
  if (reductionTileSize > 0) {
    ss << " r" << reductionTileSize;
  }
  return str;
}

std::string TilingEvaluation::toString() const {
  switch (errorStatus) {
    case EvalError::Success:
      break;
    case EvalError::MissingOrInvalidLayout:
      return "Missing or invalid layout";
    case EvalError::InvalidTilingConfig:
      return "Invalid tiling configuration";
    case EvalError::MissingTensorElementSize:
      return "Missing tensor element size";
    case EvalError::ConcatDimensionIsTiled:
      return "Concat dimension is tiled";
  }
  std::string str;
  llvm::raw_string_ostream ss(str);
  ss << "blocks: " << blockCount << ", "
     << "storage: " << tileStorageBytes << ", "
     << "accessCount: " << memoryAccessCount << ", "
     << "cacheLines: " << memoryAccessCacheLines << ", "
     << "mmaOps: " << mmaOperationCount;
  return str;
}

TilingEvaluation TilingEvaluator::evaluate(
    GraphOp graph, const TilingConfig& tilingConfig) const {
  // Get the layout of the main iteration space.
  auto inputLayout = getInputLayout(graph);
  auto outputLayout = getOutputLayout(graph).reshape(inputLayout.getShape());
  if (!inputLayout || !outputLayout) {
    return TilingEvaluation{
        .errorStatus = TilingEvaluation::EvalError::MissingOrInvalidLayout,
    };
  }

  // Verify the tiling configuration (rank and sizes).
  if (!tilingConfig.validate() ||
      tilingConfig.tileShape.size() != inputLayout.getShape().size()) {
    return TilingEvaluation{
        .errorStatus = TilingEvaluation::EvalError::InvalidTilingConfig,
    };
  }

  // Get the input element sizes.
  SmallVector<size_t> inputElementSizes;
  auto getTypeSize = [](Type type) {
    return std::max(type.getIntOrFloatBitWidth() / 8, 1u);
  };
  for (BlockArgument arg : graph.getBody()->getArguments()) {
    auto type = cast<TensorType>(arg.getType()).getElementType();
    inputElementSizes.push_back(getTypeSize(type));
  }

  // Evaluate the inputs.
  TilingEvaluation inputEval =
      evaluateImpl(inputLayout, tilingConfig, inputElementSizes, archInfo);

  // Get the output element sizes.
  SmallVector<size_t> outputElementSizes;
  for (size_t idx : llvm::seq(graph.getNumResults())) {
    auto type = cast<TensorType>(graph.getResultTypes()[idx]).getElementType();
    outputElementSizes.push_back(getTypeSize(type));
  }

  // Evaluate the outputs.
  TilingEvaluation outputEval =
      evaluateImpl(outputLayout, tilingConfig, outputElementSizes, archInfo);

  // Calculate the number of execution blocks.
  int64_t blockCount = 1;
  for (auto [size, tileSize] :
       llvm::zip_equal(inputLayout.getShape(), tilingConfig.tileShape)) {
    if (size == ShapedType::kDynamic) {
      size = kDynamicSizeValue;
    }
    blockCount *= llvm::divideCeil(size, tileSize);
  }

  // Merge the results.
  return TilingEvaluation{
      blockCount,
      std::max(inputEval.tileStorageBytes, outputEval.tileStorageBytes),
      inputEval.memoryAccessCount + outputEval.memoryAccessCount,
      inputEval.memoryAccessCacheLines + outputEval.memoryAccessCacheLines,
      inputEval.mmaOperationCount,
  };
}

int TilingEvaluator::estimateOccupancy(const TilingEvaluation& evaluation,
                                       int warpCount) const {
  // Maximum SM occupancy based on hardware limits.
  int blockOccupancy =
      std::min(archInfo.maxBlocksPerSM, archInfo.maxWarpsPerSM / warpCount);

  // Estimate the register usage per thread.
  int threadCount = warpCount * kWarpSize;
  int registersLimit =
      std::min(kMaxRegistersPerSM / threadCount, kMaxRegistersPerThread);
  int registersUsage =
      std::min(evaluation.tileStorageBytes / (threadCount * kRegisterSize),
               static_cast<int64_t>(registersLimit));
  int registerOccupancy = kMaxRegistersPerSM / (threadCount * registersUsage);

  // Return the estimated occupancy.
  return std::min(blockOccupancy, registerOccupancy);
}

// Get the input layout from the iteration space attribute.
LayoutSourceAttrInterface getInputLayout(GraphOp graph) {
  return graph.getBody()
      ->getTerminator()
      ->getAttrOfType<LayoutSourceAttrInterface>(
          TensorIRDialect::getIterationSpaceAttrName());
}

// Get the output layout for the result tensor.
LayoutSourceAttrInterface getOutputLayout(GraphOp graph) {
  SmallVector<LayoutSourceAttrInterface> outputLayouts;
  for (size_t idx : llvm::seq(graph.getNumResults())) {
    auto resultTy = cast<TensorType>(graph.getResultTypes()[idx]);
    std::optional<tcutegen::Stride> stride;
    if (auto strideAttr = graph.getResultAttrOfType<StringAttr>(
            idx, TensorIRDialect::getStrideAttrName())) {
      stride = tcutegen::from_string<tcutegen::Stride>(strideAttr.str());
    }
    auto cuteLayout = stride.has_value()
                          ? tcutegen::Layout(getShapeRef(resultTy), *stride)
                          : tcutegen::Layout(getShapeRef(resultTy));
    auto tensorSource =
        TensorSourceAttr::get(graph.getContext(), /*tensorId=*/idx,
                              /*offset=*/0, cuteLayout.toString(),
                              /*dynamicValueMapping=*/{});
    outputLayouts.push_back(tensorSource);
  }
  return outputLayouts[0];
}

}  // namespace tiling_analysis
}  // namespace mlir::nv_tensor_ir
