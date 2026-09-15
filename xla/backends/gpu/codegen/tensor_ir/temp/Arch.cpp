#include "xla/backends/gpu/codegen/tensor_ir/temp/Arch.h"

namespace mlir::nv_tensor_ir {
namespace tiling_analysis {

// Ampere A100
static const ArchInfo archInfoSm80 = {
    .numSMs = 108,
    .maxBlocksPerSM = 32,
    .maxWarpsPerSM = 64,
    .maxSharedMemoryPerSM = 164,
    .tensorCoreVersions = kMMAv2,
};

// Hopper H100
static const ArchInfo archInfoSm90 = {
    .numSMs = 132,
    .maxBlocksPerSM = 32,
    .maxWarpsPerSM = 64,
    .maxSharedMemoryPerSM = 228,
    .tensorCoreVersions = kMMAv2 | kMMAv3,
};

// Blackwell B200
static const ArchInfo archInfoSm100 = {
    .numSMs = 148,
    .maxBlocksPerSM = 32,
    .maxWarpsPerSM = 64,
    .maxSharedMemoryPerSM = 228,
    .tensorCoreVersions = kMMAv2 | kMMAv5,
};

// Generic GPU
static const ArchInfo archInfoGeneric = {
    .numSMs = 100,
    .maxBlocksPerSM = 16,
    .maxWarpsPerSM = 48,
    .maxSharedMemoryPerSM = 100,
    .tensorCoreVersions = 0,
};

/**
 * @brief Get the architecture information for a given compute capability.
 * @param cc Compute capability.
 */
const ArchInfo& getArchInfo(ComputeCapability cc) {
  switch (cc) {
    case ComputeCapability::Sm80:
      return archInfoSm80;
    case ComputeCapability::Sm90:
      return archInfoSm90;
    case ComputeCapability::Sm100:
      return archInfoSm100;
    default:
      return archInfoGeneric;
  }
}

}  // namespace tiling_analysis
}  // namespace mlir::nv_tensor_ir
