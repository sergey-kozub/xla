# Tiling analysis will live in the TensorIR repository.
cc_library(
    name = "tiling_analysis",
    hdrs = ["Arch.h", "Enumerate.h", "Evaluate.h"],
    srcs = ["Arch.cpp", "Enumerate.cpp", "Evaluate.cpp"],
    visibility = [
        "//xla/backends/gpu/autotuner:__subpackages__",
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@tensor_ir//:NVTensorIRDialect",
    ],
)
