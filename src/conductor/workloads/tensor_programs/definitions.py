
import tvm
from tvm import topi
import undecorated

tensor_programs = {
    "batch_matmul_tensorcore": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.batch_matmul_tensorcore),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_batch_matmul_tensorcore),
            "args": lambda x_dims, y_dims: (
                tvm.te.placeholder(x_dims),
                tvm.te.placeholder(y_dims)
            ),
            "default_input": ((8, 16, 64), (8, 128, 64))
        },
    },
    "batch_matmul": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.batch_matmul),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_batch_matmul),
            "args": lambda x_dims, y_dims, dtype: (
                tvm.te.placeholder(x_dims, dtype=dtype),
                tvm.te.placeholder(y_dims, dtype=dtype)
            ),
            "default_input": ((1, 5, 3), (1, 6, 3), "float32")
        },
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.batch_matmul),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_batch_matmul),
            "args": lambda x_dims, y_dims, dtype: (
                tvm.te.placeholder(x_dims, dtype=dtype),
                tvm.te.placeholder(y_dims, dtype=dtype)
            ),
            "default_input": ((1, 5, 3), (1, 6, 3), "float32")
        }
    },
    "conv1d_transpose_ncw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv1d_transpose_ncw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv1d_transpose_ncw),
            "args": lambda data_dims, kernel_dims, stride, padding, dtype, output_padding: (
                tvm.te.placeholder(data_dims, dtype=dtype),
                tvm.te.placeholder(kernel_dims, dtype=dtype),
                (stride),
                padding,
                dtype,
                output_padding
            ),
            "default_input": ((1, 3, 24), (3, 7, 5), 2, (1, 1), "float32", (1, 1))
        }
    },
    "conv1d_ncw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv1d_ncw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv1d_ncw),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dilation
            ),
            "default_input": ((1, 3, 24), (7, 3, 5), 2, (1, 1), 1)
        }
    },
    "conv1d_nwc": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv1d_nwc),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv1d_nwc),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dilation
            ),
            "default_input": ((1, 24, 3), (5, 3, 7), 2, (1, 1), 1)
        }
    },
    "conv2d_hwcn": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_hwcn),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_hwcn),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dilation
            ),
            "default_input": ((24, 24, 3, 1), (5, 5, 3, 7), (2, 2), (1, 1, 1, 1), (1, 1))
        }
    },
    "conv2d_NCHWc_int8": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_NCHWc_int8),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_NCHWc_int8),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, layout, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                layout,
                dtype
            ),
            "default_input": ((1, 4, 24, 24), (8, 4, 5, 5), (2, 2), (1, 1), (1, 1), "NCHW")
        },
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.conv2d_NCHWc),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_conv2d_NCHWc),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_layout, layout, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                in_layout,
                layout,
                dtype
            ),
            "default_input": ((1, 4, 24, 24), (8, 4, 5, 5), (2, 2), (1, 1), (1, 1), "NCHW")
        }
    },
    "conv2d_nhwc_tensorcore": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_nhwc_tensorcore),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_nhwc_tensorcore),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                (stride),
                padding,
                dilation,
                dtype
            ),
            "default_input": ((16, 24, 24, 32), (5, 5, 1, 32), (2, 2), (1, 1, 1, 1), (1, 1), "int8")
        }
    },
    "conv2d_nhwc_winograd_direct": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_nhwc_winograd_direct),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_nhwc_winograd_direct),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 24, 24, 3), (5, 5, 0, 7), (1, 1), (1, 1, 1, 1), (1, 1), "float32")
        }
    },
    "conv2d_nhwc_winograd_tensorcore": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_nhwc_winograd_tensorcore),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_nhwc_winograd_tensorcore),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 24, 24, 3), (5, 5, 0, 7), (1, 1), (1, 1, 1, 1), (1, 1), "float32")
        }
    },
    "conv2d_transpose_nchw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_transpose_nchw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_transpose_nchw),
            "args": lambda data_dims, kernel_dims, stride, padding, dtype, output_padding: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                stride,
                padding,
                dtype,
                output_padding
            ),
            "default_input": ((1, 3, 24, 24), (3, 7, 5, 5), (2, 2), 1, "float32", (1, 1))
        }
    },
    "conv2d_nchw_winograd": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_nchw_winograd),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_nchw_winograd),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 24, 24, 3), (5, 5, 0, 7), (1, 1), (1, 1, 1, 1), (1, 1), "float32")
        }
    },
    "conv2d_nchw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv2d_nchw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv2d_nchw),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                stride,
                padding,
                dilation
            ),
            "default_input": ((1, 3, 24, 24), (7, 3, 5, 5), (2, 2), (1, 1, 1, 1), (1, 1))
        }
    },
    "conv2d_nhwc": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.gpu.conv2d_nhwc),
            "schedule_gen_func": undecorated.undecorated(topi.gpu.schedule_conv2d_nhwc),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                stride,
                padding,
                dilation
            ),
            "default_input": ((1, 24, 24, 3), (7, 5, 5, 3), (2, 2), (1, 1, 1, 1), (1, 1))
        }
    },
    "conv3d_ndhwc_tensorcore": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv3d_ndhwc_tensorcore),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv3d_ndhwc_tensorcore),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 3, 24, 24, 3), (5, 5, 5, 3, 7), (2, 2, 2), 1, (1, 1, 1), "float32")
        }
    },
    "conv3d_transpose_ncdhw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv3d_transpose_ncdhw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv3d_transpose_ncdhw),
            "args": lambda data_dims, kernel_dims, stride, padding, dtype, output_padding: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dtype,
                output_padding
            ),
            "default_input": ((1, 3, 3, 24, 24), (3, 7, 5, 5, 5), (2, 2, 2), 1, "float32", (1, 1, 1))
        }
    },
    "conv3d_ncdhw_winograd": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv3d_ncdhw_winograd),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv3d_ncdhw_winograd),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 3, 24, 24, 3), (5, 5, 5, 3, 7), (2, 2, 2), 1, (1, 1, 1), "float32")
        }
    },
    "conv3d_ncdhw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv3d_ncdhw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv3d_ncdhw),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 3, 3, 24, 24), (7, 3, 5, 5, 5), (2, 2, 2), (1, 1, 1), (1, 1, 1), "float32")
        },
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.conv3d_ncdhw),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_conv3d_ncdhw),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 3, 3, 24, 24), (7, 3, 5, 5, 5), (2, 2, 2), (1, 1, 1), (1, 1, 1), "float32")
        }
    },
    "conv3d_ndhwc": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.conv3d_ndhwc),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_conv3d_ndhwc),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 3, 24, 24, 3), (5, 5, 5, 3, 7), (2, 2, 2), 1, (1, 1, 1), "float32")
        },
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.conv3d_ndhwc),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_conv3d_ndhwc),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                (stride),
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 3, 24, 24, 3), (5, 5, 5, 3, 7), (2, 2, 2), 1, (1, 1, 1), "float32")
        }
    },
    "correlation_nchw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.correlation_nchw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_correlation_nchw),
            "args": lambda data1_dims, data2_dims, kernel_size, max_displacement, stride1, stride2, padding, is_multiply: (
                tvm.te.placeholder(data1_dims, dtype="float32"),
                tvm.te.placeholder(data2_dims, dtype="float32"),
                kernel_size,
                max_displacement,
                stride1,
                stride2,
                padding,
                is_multiply
            ),
            "default_input": ((1, 3, 24, 24), (1, 3, 24, 24), 5, 2, 1, 1, (1, 1, 1, 1), False)
        }
    },
    "deformable_conv2d_nchw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.deformable_conv2d_nchw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_deformable_conv2d_nchw),
            "args": lambda data_dims, offset_dims, kernel_dims, stride, padding, dilation, deformable_groups, groups, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(offset_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                stride,
                padding,
                dilation,
                deformable_groups,
                groups,
                dtype
            ),
            # TODO: this is wrong but we will figure it out
            "default_input": ((1, 3, 24, 24), (1, 20, 24, 24), (7, 3, 5, 5), (2, 2), (1, 1), (1, 1), 1, 1, "float32")
        }
    },
    "dense_tensorcore": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.dense_tensorcore),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_dense_tensorcore),
            "args": lambda data_dims, weight_dims, bias, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(weight_dims, dtype="float32"),
                bias,
                dtype
            ),
            "default_input": ((1, 24), (48, 24), "float32")
        }
    },
    "dense_small_batch": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.gpu.dense_small_batch),
            "schedule_gen_func": undecorated.undecorated(topi.gpu.schedule_dense_small_batch),
            "args": lambda data_dims, weight_dims, bias, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(weight_dims, dtype="float32"),
                bias,
                dtype
            ),
            "default_input": ((1, 24), (48, 24), "float32")
        }
    },
    "dense_large_batch": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.gpu.dense_large_batch),
            "schedule_gen_func": undecorated.undecorated(topi.gpu.schedule_dense_large_batch),
            "args": lambda data_dims, weight_dims, bias, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(weight_dims, dtype="float32"),
                bias,
                dtype
            ),
            "default_input": ((16, 24), (48, 24), "float32")
        }
    },
    "dense_int8": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.dense_int8),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_dense_int8),
            "args": lambda data_dims, weight_dims, bias, dtype: (
                tvm.te.placeholder(data_dims, dtype="int8"),
                tvm.te.placeholder(weight_dims, dtype="int8"),
                bias,
                dtype
            ),
            "default_input": ((16, 24), (48, 24), "int32")
        }
    },
    "depthwise_conv2d_nchw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.depthwise_conv2d_nchw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_depthwise_conv2d_nchw),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                stride,
                padding,
                dilation,
                dtype
            ),
            "default_input": ((1, 3, 24, 24), (3, 2, 5, 5), (1, 1), 1, (1, 1), "float32")
        }
    },
    "group_conv2d_nchw": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.group_conv2d_nchw),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_group_conv2d_nchw),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, groups, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(kernel_dims, dtype="float32"),
                stride,
                padding,
                dilation,
                groups,
                dtype
            ),
            "default_input": ((1, 3, 24, 24), (9, 3, 5, 5), (1, 1), (1, 1, 1, 1), (1, 1), 3, "float32")
        }
    },
    "group_conv2d_NCHWc_int8": {
        "cuda": {
            "compute_gen_func": undecorated.undecorated(topi.cuda.group_conv2d_NCHWc_int8),
            "schedule_gen_func": undecorated.undecorated(topi.cuda.schedule_group_conv2d_NCHWc_int8),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, groups, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                groups,
                dtype
            ),
            "default_input": ((1, 4, 24, 24), (8, 4, 5, 5), (2, 2), (1, 1), (1, 1), "NCHW")
        }
    },
    "bitserial_conv2d_nchw": {
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.bitserial_conv2d_nchw),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_bitserial_conv2d_nchw),
            "args": lambda data_dims, kernel_dims, stride, padding, in_bits, weight_bits, pack_dtype, out_dtype, unipolar: (
                tvm.te.placeholder(data_dims, dtype=pack_dtype),
                tvm.te.placeholder(kernel_dims, dtype=pack_dtype),
                stride,
                padding,
                in_bits,
                weight_bits,
                pack_dtype,
                out_dtype,
                unipolar
            ),
            "default_input": ((1, 56, 56, 64), (3, 3, 64, 64), 1, 1, 2, 1, "uint32", "int32", True)
        }
    },
    "bitserial_conv2d_nhwc": {
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.bitserial_conv2d_nhwc),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_bitserial_conv2d_nhwc),
            "args": lambda data_dims, kernel_dims, stride, padding, in_bits, weight_bits, pack_dtype, out_dtype, unipolar: (
                tvm.te.placeholder(data_dims, dtype=pack_dtype),
                tvm.te.placeholder(kernel_dims, dtype=pack_dtype),
                stride,
                padding,
                in_bits,
                weight_bits,
                pack_dtype,
                out_dtype,
                unipolar
            ),
            "default_input": ((1, 56, 56, 64), (3, 3, 64, 64), 1, 1, 2, 1, "uint32", "int32", True)
        }
    },
    "bitserial_dense": {
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.bitserial_dense),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_bitserial_dense),
            "args": lambda data_dims, kernel_dims, in_bits, weight_bits, pack_dtype, out_dtype, unipolar: (
                tvm.te.placeholder(data_dims, dtype=pack_dtype),
                tvm.te.placeholder(kernel_dims, dtype=pack_dtype),
                in_bits,
                weight_bits,
                pack_dtype,
                out_dtype,
                unipolar
            ),
            "default_input": ((1, 56, 56, 64), (3, 3, 64, 64), 1, 1, 2, 1, "uint32", "int32", True)
        }
    },
    "dense_nopack": {
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.dense_nopack),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_dense_nopack),
            "args": lambda data_dims, weight_dims, bias, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(weight_dims, dtype="float32"),
                bias,
                dtype
            ),
            "default_input": ((1, 24), (48, 24), "float32")
        }
    },
    "dense_pack": {
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.dense_pack),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_dense_pack),
            "args": lambda data_dims, weight_dims, bias, dtype: (
                tvm.te.placeholder(data_dims, dtype="float32"),
                tvm.te.placeholder(weight_dims, dtype="float32"),
                bias,
                dtype
            ),
            "default_input": ((1, 24), (48, 24), "float32")
        }
    },
    "depthwise_conv2d_NCHWc": {
        "x86": {
            "compute_gen_func": undecorated.undecorated(topi.x86.depthwise_conv2d_NCHWc),
            "schedule_gen_func": undecorated.undecorated(topi.x86.schedule_depthwise_conv2d_NCHWc),
            "args": lambda data_dims, kernel_dims, stride, padding, dilation, in_layout, layout, in_dtype, dtype: (
                tvm.te.placeholder(data_dims, dtype=in_dtype),
                tvm.te.placeholder(kernel_dims, dtype=in_dtype),
                stride,
                padding,
                dilation,
                in_layout,
                layout,
                dtype
            ),
            "default_input": ((1, 3, 24, 24), (3, 2, 5, 5), (1, 1), 1, (1, 1), "float32")
        }
    }
}