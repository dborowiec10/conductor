
from conductor.workloads.tensor_programs.inputs import inp_conv1d_transpose_ncw, \
    inp_for_conv1d, inp_for_conv2d, inp_for_conv2d_transpose, inp_for_conv2d_nchw_nhwc, \
    inp_for_conv3d_nchw_nhwc_t, inp_for_conv3d_nchw_nhwc, inp_for_correlation, \
    inp_for_deformable, inp_for_dense, inp_for_depthwise_conv2d, inp_for_group_conv2d, \
    inp_bitserial_conv2d, inp_bitserial_dense, inp_for_conv2d_NCHWc_int8, inp_for_conv2d_nhwc, \
    inp_for_conv2d_nchw_wino, inp_for_conv3d_tensor, inp_for_conv3d_wino, \
    inp_for_group_conv2d_NCHWc_int8, inp_for_depthwise_conv2d_NCHWc

tensor_programs = [
    ("batch_matmul", ((1, 5, 3), (1, 6, 3), "float32")),
    ("batch_matmul", ((5, 16, 32), (5, 48, 32), "float32")),
    ("batch_matmul", ((5, 1024, 32), (5, 2048, 32), "float32")),
    ("conv1d_transpose_ncw", inp_conv1d_transpose_ncw(1, 3, 224, 32, 5, 1, 0, "float32", 0)),
    ("conv1d_transpose_ncw", inp_conv1d_transpose_ncw(1, 1, 1024, 1, 512, 1, 256, "float32", 0)),
    ("conv1d_transpose_ncw", inp_conv1d_transpose_ncw(1, 1, 10, 1, 5, 1, (2,3), "float32", 0)),
    ("conv1d_ncw", inp_for_conv1d(1, 1, 8, 1, 3, 1, 1, 'VALID', "float32", "NCW")),
    ("conv1d_ncw", inp_for_conv1d(1, 16, 32, 16, 1, 1, 1, 'SAME', "float32", "NCW")),
    ("conv1d_ncw", inp_for_conv1d(1, 5, 27, 18, 3, 1, 1, 'VALID', "float32", "NCW")),
    ("conv1d_nwc", inp_for_conv1d(1, 1, 8, 1, 3, 1, 1, 'VALID', "float32", "NWC")),
    ("conv1d_nwc", inp_for_conv1d(1, 16, 32, 16, 1, 1, 1, 'SAME', "float32", "NWC")),
    ("conv1d_nwc", inp_for_conv1d(1, 5, 27, 18, 3, 1, 1, 'VALID', "float32", "NWC")),
    ("conv2d_hwcn", inp_for_conv2d(1, 256, 32, 256, 3, 1, "SAME", 1)),
    ("conv2d_hwcn", inp_for_conv2d(4, 128, 16, 256, 5, 2, "VALID", 1)),
    ("conv2d_hwcn", inp_for_conv2d(1, 256, 32, 256, 3, 1, "SAME", 2)),
    ("conv2d_transpose_nchw", inp_for_conv2d_transpose(1, 3, (224, 224), 1, (1, 1), (1, 1), (0, 0, 0, 0), "float32", (0, 0))),
    ("conv2d_transpose_nchw", inp_for_conv2d_transpose(1, 3, (224, 224), 32, (2, 2), (2, 2), (0, 0, 0, 0), "float32", (1, 1))),
    ("conv2d_transpose_nchw", inp_for_conv2d_transpose(3, 64, (8, 8), 64, (25, 25), (2, 1), (8, 0, 7, 0), "float32", (1, 0))),
    ("conv2d_nchw", inp_for_conv2d_nchw_nhwc(1, 64, 56, 64, 3, 1, 1, 2, "NCHW")),
    ("conv2d_nchw", inp_for_conv2d_nchw_nhwc(2, 2, 2, 2, 2, 2, 2, 1, "NCHW")),
    ("conv2d_nchw", inp_for_conv2d_nchw_nhwc(1, 2048,  10, 126, 3, 1, 1, 1, "NCHW")),
    ("conv2d_nchw", inp_for_conv2d_nchw_nhwc(1, 512,   19,  64,  1, 1, "SAME", 1, "NCHW")),
    ("conv2d_nhwc", inp_for_conv2d_nchw_nhwc(4, 128, 16, 128, 5, 2, "SAME", 1, "NHWC")),
    ("conv2d_nhwc", inp_for_conv2d_nchw_nhwc(1, 128, 16, 256, 3, 2, (0, 0, 1, 1), 1, "NHWC")),
    ("conv2d_nhwc", inp_for_conv2d_nchw_nhwc(1, 256, 32, 256, 3, 1, (1, 1, 2, 2), 2, "NHWC")),
    ("conv3d_transpose_ncdhw", inp_for_conv3d_nchw_nhwc_t(1, 3, (24, 24, 24), 1, (1, 1, 1), (1, 1, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0), "float32")),
    ("conv3d_transpose_ncdhw", inp_for_conv3d_nchw_nhwc_t(1, 3, (24, 24, 24), 16, (3, 3, 3), (1, 1, 1), (0, 0, 0, 0, 0, 0), (0, 0, 0), "float32")),
    ("conv3d_transpose_ncdhw", inp_for_conv3d_nchw_nhwc_t(1, 8, (32, 32, 32), 64, (5, 5, 5), (2, 2, 2), (1, 1, 1, 1, 1, 1), (1, 1, 1), "float32")),
    ("conv3d_ncdhw", inp_for_conv3d_nchw_nhwc(1, 16, 32, 16, 3, 1, "SAME", 1, "float32", "NCDHW")),
    ("conv3d_ncdhw", inp_for_conv3d_nchw_nhwc(1, 1, (20, 256, 256), 32, (1, 3, 3), (1, 2, 2), "SAME", 1, "float32", "NCDHW")),
    ("conv3d_ncdhw", inp_for_conv3d_nchw_nhwc(1, 4, (20, 256, 256), 8, (1, 5, 5), (1, 2, 2), (0, 2, 2), 1, "float32", "NCDHW")),
    ("conv3d_ndhwc", inp_for_conv3d_nchw_nhwc(1, 32, 32, 5, 1, 1, 0, 1, "float32", "NDHWC")),
    ("conv3d_ndhwc", inp_for_conv3d_nchw_nhwc(1, 32, 32, 5, 1, 1, (0, 0, 0, 1, 1, 1), 1, "float32", "NDHWC")),
    ("conv3d_ndhwc", inp_for_conv3d_nchw_nhwc(1, 32, 32, 1, 1, 1, (2, 1, 0), 1, "float32", "NDHWC")),
    ("correlation_nchw", inp_for_correlation((1, 3, 10, 10), 1, 4, 1, 1, 4, True)),
    ("deformable_conv2d_nchw", inp_for_deformable(1, 16, 7, 16, 1, 1, 0, 1, 4, 1, "float32")),
    ("deformable_conv2d_nchw", inp_for_deformable(1, 16, 7, 16, 3, 1, 2, 2, 1, 1, "float32")),
    ("dense_nopack", inp_for_dense(2, 256, 1000, "int32")),
    ("dense_nopack", inp_for_dense(9, 2048, 5000, "int32")),
    ("dense_pack", inp_for_dense(2, 256, 1000, "int32")),
    ("dense_pack", inp_for_dense(9, 2048, 5000, "int32")),
    ("dense_small_batch", inp_for_dense(1, 1024, 1000, "float32")),
    ("dense_large_batch", inp_for_dense(4096, 1024, 1000, "float32")),
    ("dense_int8", inp_for_dense(2, 256, 1000, "int32")),
    ("dense_int8", inp_for_dense(9, 2048, 5000, "int32")),
    ("depthwise_conv2d_nchw", inp_for_depthwise_conv2d(1, 32, 112, 1, 3, 1, "SAME", 1, "float32")),
    ("depthwise_conv2d_nchw", inp_for_depthwise_conv2d(1, 728, 64, 1, 3, 1, "SAME", 2, "float32")),
    ("group_conv2d_nchw", inp_for_group_conv2d(1, 128, 56, 128, 3, 1, 1, 1, 32, "float32")),
    ("group_conv2d_nchw", inp_for_group_conv2d(1, 1024, 7, 1024, 3, 1, 1, 1, 32, "float32")),
    ("bitserial_conv2d_nchw", inp_bitserial_conv2d(1, 56, 64, 64, 3, 1, 1, 1, 1, True, "uint32", "int32", "NCHW")),
    ("bitserial_conv2d_nchw", inp_bitserial_conv2d(1, 56, 64, 64, 3, 1, 1, 2, 2, False, "uint32", "int32", "NCHW")),
    ("bitserial_conv2d_nhwc", inp_bitserial_conv2d(1, 56, 64, 64, 3, 1, 1, 1, 1, True, "uint32", "int32", "NHCW")),
    ("bitserial_conv2d_nhwc", inp_bitserial_conv2d(1, 56, 64, 64, 3, 1, 1, 2, 2, False, "uint32", "int32", "NHCW")),
    ("bitserial_dense", inp_bitserial_dense(1, 1024, 1000, 1, 1, True, "uint32", "int16")),
    ("bitserial_dense", inp_bitserial_dense(1, 1024, 1000, 2, 1, False, "uint32", "int16")),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(1, 64, 56, 64, 3, 1, 1, 1, "int8", "int32", "x86"), "x86"),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(1, 64, 56, 64, 3, 1, 1, 1, "int8", "int32", "cuda"), "cuda"),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(4, 4, 4, 4, 4, 4, 4, 1, "int8", "int32", "x86"), "x86"),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(4, 4, 4, 4, 4, 4, 4, 1, "int8", "int32", "cuda"), "cuda"),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(1, 512,   19,  32,  1, 1, "SAME", 1, "int8", "int32", "x86"), "x86"),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(1, 512,   19,  32,  1, 1, "SAME", 1, "int8", "int32", "cuda"), "cuda"),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(1,  32,    8,  32,  3, 1, (1, 2, 2, 1), 1, "int8", "int32", "x86"), "x86"),
    ("conv2d_NCHWc_int8", inp_for_conv2d_NCHWc_int8(1,  32,    8,  32,  3, 1, (1, 2, 2, 1), 1, "int8", "int32", "cuda"), "cuda"),
    ("conv2d_nhwc_winograd_direct", inp_for_conv2d_nhwc(1,  64, 56,  64, 3, 1, 1,  1, "float32", "float32")),
    ("conv2d_nchw_winograd", inp_for_conv2d_nchw_wino(1,  64, 56,  64, 3, 1, 1,  1, "float32", "float32")),
    ("conv3d_ncdhw_winograd", inp_for_conv3d_wino(1, 61, 20, 120, 3, 3, 1, 0, 1, "float32", "float32", "NCDHW")),
    ("conv3d_ncdhw_winograd", inp_for_conv3d_wino(1, 64, 12, 128, 1, 3, 1, 1, 1, "float32", "float32", "NCDHW")),
    ("group_conv2d_NCHWc_int8", inp_for_group_conv2d_NCHWc_int8(1, 256, 3, 224, 64, 6, 2, 3, 1, "int8", "int32")),
    ("group_conv2d_NCHWc_int8", inp_for_group_conv2d_NCHWc_int8(1, 256, 6, 224, 512, 6, 2, 3, 1, "int8", "int32"))
]