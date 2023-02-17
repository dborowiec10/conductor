# Definitions of functions that organise inputs for TVM TOPI operators


def inp_conv1d_transpose_ncw(batch, in_channel, inp_width, num_filter, kernel_size, stride, padding, dtype, out_padding):
    inp_shape = [batch, in_channel, inp_width]
    kern_shape = [in_channel, num_filter, kernel_size]
    return (inp_shape, kern_shape, stride, padding, dtype, out_padding)


def inp_for_conv1d(batch, in_channels, in_width, filters, kernel_size, stride, dilation, padding, dtype, layout):
    if layout == 'NCW':
        in_shape = [batch, in_channels, in_width]
        kernel_shape = [filters, in_channels, kernel_size]
    else:
        in_shape = [batch, in_width, in_channels]
        kernel_shape = [kernel_size, in_channels, filters]
    return (in_shape, kernel_shape, stride, padding, dilation), (batch, in_channels, in_width, filters, kernel_size, stride, dilation, padding, dtype, layout)


def inp_for_conv2d(batch, in_channels, in_width, filters, kernel_size, stride, padding, dilation):
    in_shape = [in_width, in_width, in_channels, batch]
    kernel_shape = [kernel_size, kernel_size, in_channels, filters]
    return (in_shape, kernel_shape, stride, padding, dilation), (batch, in_channels, in_width, filters, kernel_size, stride, padding, dilation)


def inp_for_conv2d_transpose(batch, in_channels, in_size, filters, kernel_size, stride, padding, dtype, output_padding):
    in_width, in_height = in_size
    kern_width, kern_height = kernel_size
    in_shape = [batch, in_channels, in_height, in_width]
    kernel_shape = [in_channels, filters, kern_height, kern_width]
    return (in_shape, kernel_shape, stride, padding, dtype, output_padding), (batch, in_channels, in_size, filters, kernel_size, stride, padding, dtype, output_padding)


def inp_for_conv2d_nchw_nhwc(batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, layout):
    in_width = in_height = in_size
    kern_width = kern_height = kernel_size

    if layout == "NCHW":
        in_shape = [batch, in_channel, in_height, in_width]
        kernel_shape = [filters, in_channel, kern_height, kern_width]
    elif layout == "NHWC":
        in_shape = [batch, in_height, in_width, in_channel]
        kernel_shape = [kern_height, kern_width, in_channel, filters]
    return (in_shape, kernel_shape, stride, padding, dilation), (batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, layout)


def inp_for_conv3d_nchw_nhwc_t(batch, in_channel, in_size, filters, kernel_size, stride, padding, output_padding, dtype):
    in_depth, in_height, in_width = in_size
    kernel_depth, kernel_height, kernel_width = kernel_size
    in_shape = (batch, in_channel, in_depth, in_height, in_width)
    kernel_shape = (in_channel, filters, kernel_depth,
                    kernel_height, kernel_width)
    return (in_shape, kernel_shape, stride, padding, dtype, output_padding), (batch, in_channel, in_size, filters, kernel_size, stride, padding, output_padding, dtype)


def inp_for_conv3d_nchw_nhwc(batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, dtype, layout):
    if isinstance(in_size, tuple):
        in_depth, in_height, in_width = in_size
    else:
        in_depth = in_height = in_width = in_size
    if isinstance(kernel_size, tuple):
        kernel_depth, kernel_height, kernel_width = kernel_size
    else:
        kernel_depth = kernel_height = kernel_width = kernel_size

    if layout == "NDHWC":
        in_shape = [batch, in_depth, in_height, in_width, in_channel]
        kernel_shape = [kernel_depth, kernel_height,
                        kernel_width, in_channel, filters]
    elif layout == "NCDHW":
        in_shape = [batch, in_channel, in_depth, in_height, in_width]
        kernel_shape = [filters, in_channel,
                        kernel_depth, kernel_height, kernel_width]
    return (in_shape, kernel_shape, stride, padding, dilation, dtype), (batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, dtype, layout)


def inp_for_correlation(data, kernel, max_disp, stride1, stride2, pad_size, is_multiply):
    data1 = data2 = data
    return (data1, data2, kernel, max_disp, stride1, stride2, pad_size, is_multiply), (data, kernel, max_disp, stride1, stride2, pad_size, is_multiply)


def inp_for_deformable(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, def_groups, groups, dtype):
    a_shape = [batch, in_channel, in_size, in_size]
    out_size = (in_size - (kernel - 1) * dilation -
                1 + 2 * padding) // stride + 1
    offset_shape = [batch, def_groups *
                    kernel * kernel * 2, out_size, out_size]
    w_shape = [num_filter, in_channel, kernel, kernel]
    return (a_shape, offset_shape, w_shape, stride, padding, dilation, def_groups, groups, dtype), (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, def_groups, groups, dtype)


def inp_for_dense(batch, in_dims, out_dims, dtype):
    return ((batch, in_dims), (out_dims, in_dims), None, dtype), (batch, in_dims, out_dims, dtype)


def inp_for_depthwise_conv2d(batch, in_channel, in_size, channel_multiplier, kernel_size, stride, padding, dilation, dtype):
    in_width = in_height = in_size
    kernel_width = kernel_height = kernel_size
    stride_h = stride_w = stride
    in_shape = [batch, in_channel, in_height, in_width]
    kernel_shape = [in_channel, channel_multiplier,
                    kernel_height, kernel_width]
    return (in_shape, kernel_shape, (stride_h, stride_w), padding, dilation, dtype), (batch, in_channel, in_size, channel_multiplier, kernel_size, stride, padding, dilation, dtype)


def inp_for_group_conv2d(batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, groups, dtype):
    in_height = in_width = in_size
    in_shape = [batch, in_channel, in_height, in_width]
    kernel_shape = [filters, in_channel // groups, kernel_size, kernel_size]
    return (in_shape, kernel_shape, stride, padding, dilation, groups, dtype), (batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, groups, dtype)


def inp_bitserial_conv2d(batch, in_size, in_channel, num_filter, kernel, stride, padding, activ_bits, weight_bits, unipolar, in_dtype, out_dtype, layout):
    in_height = in_width = in_size
    if layout == "NCHW":
        in_shape = [batch, in_channel, in_height, in_width]
        weight_shape = [num_filter, in_channel, kernel, kernel]
    else:
        in_shape = [batch, in_height, in_width, in_channel]
        weight_shape = [kernel, kernel, in_channel, num_filter]

    return (in_shape, weight_shape, stride, padding, activ_bits, weight_bits, in_dtype, out_dtype, unipolar), (batch, in_size, in_channel, num_filter, kernel, stride, padding, activ_bits, weight_bits, unipolar, in_dtype, out_dtype, layout)


def inp_bitserial_dense(batch, in_dim, out_dim, activ_bits, weight_bits, unipolar, in_dtype, out_dtype):
    in_shape = [batch, in_dim]
    weight_shape = [out_dim, in_dim]
    return (in_shape, weight_shape, activ_bits, weight_bits, in_dtype, out_dtype, unipolar), (batch, in_dim, out_dim, activ_bits, weight_bits, unipolar, in_dtype, out_dtype)


def inp_for_conv2d_NCHWc_int8(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype, platform):
    in_height = in_width = in_size

    if platform == "x86":
        oc_block = 1
        for bn in range(16, 0, -1):
            if num_filter % bn == 0:
                oc_block = bn
                break

        ic_block = 1
        for bn in range(oc_block, 0, -1):
            if in_channel % bn == 0:
                ic_block = bn
                break
        in_shape = [batch, in_channel//ic_block, in_height, in_width, ic_block]
        weight_shape = [num_filter//oc_block, in_channel //
                        ic_block, kernel, kernel, ic_block, oc_block]
        return (in_shape, weight_shape, (stride, stride), padding, (dilation, dilation), 'NCHW%dc' % ic_block, 'NCHW%dc' % oc_block, in_dtype, dtype), (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype, platform)
    else:
        in_shape = [batch, in_channel, in_height, in_width]
        weight_shape = [num_filter, in_channel, kernel, kernel]

        return (in_shape, weight_shape, (stride, stride), padding, (dilation, dilation), 'NCHW4c', in_dtype, dtype), (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype, platform)


def inp_for_conv2d_nhwc(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype):
    in_height = in_width = in_size
    in_shape = [batch, in_height, in_width, in_channel]
    weight_shape = [kernel, kernel, in_channel, num_filter]
    return (in_shape, weight_shape, stride, padding, dilation, in_dtype, dtype), (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype)


def inp_for_conv2d_nchw_wino(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype):
    in_height = in_width = in_size
    in_shape = [batch, in_channel, in_height, in_width]
    weight_shape = [num_filter, in_channel, kernel, kernel]
    return (in_shape, weight_shape, stride, padding, dilation, in_dtype, dtype), (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype)


def inp_for_conv3d_tensor(batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, in_dtype, dtype, layout):
    if isinstance(in_size, tuple):
        in_depth, in_height, in_width = in_size
    else:
        in_depth = in_height = in_width = in_size
    if isinstance(kernel_size, tuple):
        kernel_depth, kernel_height, kernel_width = kernel_size
    else:
        kernel_depth = kernel_height = kernel_width = kernel_size
    in_shape = [batch, in_depth, in_height, in_width, in_channel]
    kernel_shape = [kernel_depth, kernel_height,
                    kernel_width, in_channel, filters]
    return (in_shape, kernel_shape, stride, padding, dilation, in_dtype, dtype), (batch, in_channel, in_size, filters, kernel_size, stride, padding, dilation, in_dtype, dtype, layout)


def inp_for_conv3d_wino(batch, in_channel, in_size, filters, kernel_size, space_kernel_size, stride, padding, dilation, in_dtype, dtype, layout):
    if isinstance(in_size, tuple):
        in_depth, in_height, in_width = in_size
    else:
        in_depth = in_height = in_width = in_size
    in_shape = [batch, in_channel, in_depth, in_height, in_width]
    kernel_shape = [filters, in_channel, kernel_size,
                    space_kernel_size, space_kernel_size]
    return (in_shape, kernel_shape, stride, padding, dilation, in_dtype, dtype), (batch, in_channel, in_size, filters, kernel_size, space_kernel_size, stride, padding, dilation, in_dtype, dtype, layout)


def inp_for_group_conv2d_NCHWc_int8(batch, in_channel, groups, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype):
    in_height = in_width = in_size

    oc_block = 1
    for bn in range(16, 0, -1):
        if num_filter % bn == 0:
            oc_block = bn
            break
    ic_block = 8
    in_shape = [batch, in_channel//ic_block, in_height, in_width, ic_block]
    weight_shape = [num_filter//oc_block, in_channel //
                    ic_block//groups, kernel, kernel, ic_block//4, oc_block]
    return (in_shape, weight_shape, stride, padding, dilation, groups, in_dtype, dtype), (batch, in_channel, groups, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype)


def inp_for_depthwise_conv2d_NCHWc(batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype):
    in_height = in_width = in_size
    oc_block = 1
    for bn in range(16, 0, -1):
        if num_filter % bn == 0:
            oc_block = bn
            break

    ic_block = 1
    for bn in range(oc_block, 0, -1):
        if in_channel % bn == 0:
            ic_block = bn
            break
    in_shape = [batch, in_channel//ic_block, in_height, in_width, ic_block]
    weight_shape = [num_filter//oc_block, in_channel //
                    ic_block, kernel, kernel, ic_block, oc_block]
    return (in_shape, weight_shape, (stride, stride), padding, (dilation, dilation), 'NCHW%dc' % ic_block, 'NCHW%dc' % oc_block, in_dtype, dtype), (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation, in_dtype, dtype)
