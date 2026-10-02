load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# Tiling analysis will live in the TensorIR repository.
cc_library(
    name = "tiling_analysis",
    srcs = [
        "Arch.cpp",
        "Enumerate.cpp",
        "Evaluate.cpp",
    ],
    hdrs = [
        "Arch.h",
        "Enumerate.h",
        "Evaluate.h",
    ],
    visibility = [
        "//xla/backends/gpu/autotuner:__subpackages__",
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@tensor_ir//:NVTensorIRDialect",
    ],
)
