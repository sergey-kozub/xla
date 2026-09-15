load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@tensor_ir//:NVTensorIRDialect",
    ],
)

cc_library(
    name = "support",
    srcs = ["support.cc"],
    hdrs = ["support.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:decision",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "compilation_pipeline",
    srcs = ["compilation_pipeline.cc"],
    hdrs = ["compilation_pipeline.h"],
    deps = [
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Pass",
        "@tensor_ir//:NVTensorIRCudaTilePipelines",
        "@tensor_ir//:NVTensorIRToCudaTileConversion",
    ],
)

cc_library(
    name = "fusion",
    srcs = ["fusion.cc"],
    hdrs = ["fusion.h"],
    visibility = [
        "//xla/backends/gpu/codegen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
    ],
    deps = [
        ":compilation_pipeline",
        ":conversion",
        ":support",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen:kernel_compiler",
        "//xla/backends/gpu/runtime:tensor_ir_kernel_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/hlo/ir:hlo",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:ir_emitter_context",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tensor_ir//:NVTensorIRCompiler",
        "@tensor_ir//:NVTensorIRDialect",
        "@tensor_ir//:NVTensorIRRuntime",
        "@tensor_ir//:NVTensorIRUtils",
    ],
)
