"""Provides the repository macro to import Tensor IR."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Tensor IR."""
    TENSOR_IR_COMMIT = "fc291f0d462426b6397825bac8e90b6aa11fb04b"
    TENSOR_IR_SHA256 = "0da37f0cdf1c33c98da1c2874db1ce8d5488a1bae45cddb8ced195e6393599ab"

    tf_http_archive(
        name = "tensor_ir",
        build_file = "//third_party/tensor_ir:tensor_ir.BUILD",
        sha256 = TENSOR_IR_SHA256,
        strip_prefix = "tensor-ir-{}".format(TENSOR_IR_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/tensor-ir/archive/{}.tar.gz".format(TENSOR_IR_COMMIT)),
        patch_file = [
            "//third_party/tensor_ir:patches/unused_variable.patch",
            "//third_party/tensor_ir:patches/symbol_op_interface.patch",
            "//third_party/tensor_ir:patches/hotfixes.patch",
        ],
    )
