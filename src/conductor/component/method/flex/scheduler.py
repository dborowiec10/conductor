import math
from functools import reduce
import tvm

from conductor.component.method.flex.model import WalkerGroup
from conductor.component.method.flex.task import conv2d_nchwc_layout
from conductor.component.method.flex.utils import flatten_graph, assert_print, Config


class Rewriter(object):
    def __init__(self, configs):
        self.graph_config= configs.graph_config
        self.op_config_lst = configs.op_config_lst

    def rewrite(self, task):
        """
        this is a hard code manner,
        we don't know how to generalize this change
        because it even need compute rewrite and schedule rewrite
        """
        assert task.target == "llvm", "Only rewrite for CPU"
        assert task.category == "conv2d"
        # schedule rewrite
        import copy
        new_graph_config = copy.deepcopy(self.graph_config)
        new_op_config_lst = copy.deepcopy(self.op_config_lst)
        # must compute inline as original config may split channel differently
        new_graph_config["inline"] = [[1, 0]]
        # fetch conv config
        conv_config = self.op_config_lst[1]
        new_config = new_op_config_lst[1]
        # change out_channel config
        vlen1 = conv_config["reduce"][0][-1]
        vlen2 = conv_config["spatial"][1][-1]
        new_config["spatial"].append([1] * len(new_config["spatial"][0]))
        new_config["spatial"][-1][-1] = vlen2
        new_config["spatial"][1][-1] = 1
        new_config["reduce"][0][-1] = 1
        new_config["reduce"].insert(1, [1] * len(new_config["reduce"][0]))
        new_config["reduce"][1][-1] = vlen1
        # compute rewrite
        kwargs = {"vlen1": vlen1, "vlen2": vlen2}
        ops, bufs = conv2d_nchwc_layout(*task.args, **kwargs)
        return ops, bufs, new_graph_config, new_op_config_lst

class Scheduler(object):
    def __init__(self, name, task, space, perf_path=None, use_model=False, rewrite=False):
        self.name = name
        self.task = task
        self.space = space
        self.rewrite = rewrite
        self.perf_path = perf_path
        self.use_model = use_model
        if task is not None and task.category is not None:
            category = task.category
        else:
            category = "none"
        self.walker_group = WalkerGroup(category + "_" + self.name, self.space)


class OpScheduler(Scheduler):
    def __init__(self, task, space, op_pos, perf_path=None, use_model=False, rewrite=False):
        Scheduler.__init__(self, "op" + str(op_pos), task, space, perf_path=perf_path, use_model=use_model, rewrite=rewrite)
        self.op_pos = op_pos

    @staticmethod
    def generate_op_schedule(target, config):
        # NOTE: COMMENTED OUT BECAUSE IS DEAD CODE (Original FlexTensor had never used it)
        # def _cuda_schedule_split_fuse(s, op, op_state):
        #     # assert_print(op in s)

        #     # always cache write here
        #     # if op.num_outputs > 1:
        #     #     raise RuntimeWarning("Too many outputs in one operation!")
        #     write_cache = s.cache_write(op.output(0), "local")

        #     # always cache read here
        #     read_cache_share_lst = []
        #     read_cache_local_lst = []
        #     for t in op.input_tensors:
        #         share = s.cache_read(t, "shared", [write_cache])
        #         read_cache_share_lst.append(share)
        #         local = s.cache_read(share, "local", [write_cache])
        #         read_cache_local_lst.append(local)

        #     # spatial split
        #     spatial_axes = s[op].op.axis
        #     splited_spatial_axes = []
        #     if "spatial" in config and len(config["spatial"]) > 0:
        #         # to align each axis
        #         assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
        #         for axis, nparts in zip(spatial_axes, config["spatial"]):
        #             tmp_buffer = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[op].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #             tmp_buffer.append(axis)
        #             splited_spatial_axes.append(tmp_buffer)
        #     else:
        #         for axis in spatial_axes:
        #             splited_spatial_axes.append([axis])
        #     assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")     # must be non-empty

        #     # always reorder and fuse here
        #     spatial_fuse_lsts = []
        #     spatial_fuse_extents = []
        #     reorder_lst = []
        #     fused_spatial_axes = []
        #     for count in range(len(splited_spatial_axes[0])):
        #         tmp_buffer = [x[count] for x in splited_spatial_axes]
        #         tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
        #         spatial_fuse_lsts.append(tmp_buffer)
        #         spatial_fuse_extents.append(tmp_extent)
        #         reorder_lst.extend(tmp_buffer)
        #     s[op].reorder(*reorder_lst)
        #     for fuse_lst in spatial_fuse_lsts:
        #         fused = s[op].fuse(*fuse_lst)
        #         fused_spatial_axes.append(fused)
        #     kernel_scope = fused_spatial_axes[0]
            
        #     # always bind here
        #     length = len(fused_spatial_axes)
        #     thread_extents = 1
        #     assert_print(length > 1, "fused axes length <= 1")
        #     if 2 <= length <= 3:
        #         s[op].bind(fused_spatial_axes[0], tvm.te.thread_axis("blockIdx.x"))
        #         s[op].bind(fused_spatial_axes[1], tvm.te.thread_axis("threadIdx.x"))
        #         thread_pos = fused_spatial_axes[1]
        #         thread_extents = spatial_fuse_extents[1]
        #     else:
        #         s[op].bind(fused_spatial_axes[0], tvm.te.thread_axis("blockIdx.x"))
        #         s[op].bind(fused_spatial_axes[1], tvm.te.thread_axis("vthread"))
        #         s[op].bind(fused_spatial_axes[2], tvm.te.thread_axis("threadIdx.x"))
        #         thread_pos = fused_spatial_axes[2]
        #         thread_extents = spatial_fuse_extents[2]

        #     # always compute at here
        #     s[write_cache].compute_at(s[op], thread_pos)

        #     # reduce_split
        #     reduced_axes = s[write_cache].op.reduce_axis
        #     splited_reduced_axes = []
        #     if "reduce" in config and len(config["reduce"]) > 0:
        #         # to align each axis
        #         assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        #         for axis, nparts in zip(reduced_axes, config["reduce"]):
        #             tmp_buffer = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[write_cache].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #             tmp_buffer.append(axis)
        #             splited_reduced_axes.append(tmp_buffer)
        #     else:
        #         for axis in reduced_axes:
        #             splited_reduced_axes.append([axis])
        #     share_pos = None
        #     local_pos = None
        #     # if has reduce axes
        #     if len(splited_reduced_axes) > 0:
        #         # always reorder here
        #         reduced_nonfuse_lsts = []
        #         reorder_lst = []
        #         length = len(splited_reduced_axes[0])
                
        #         for count in range(length):
        #             tmp_buffer = [x[count] for x in splited_reduced_axes]
        #             reduced_nonfuse_lsts.append(tmp_buffer)
        #             reorder_lst.extend(tmp_buffer)
        #         # change the order of reduce axes and spatial axes
        #         reorder_lst.extend(s[write_cache].op.axis)
        #         s[write_cache].reorder(*reorder_lst)

        #         if length == 1:
        #             share_pos = reduced_nonfuse_lsts[-1][0]
        #         else:
        #             share_pos = reduced_nonfuse_lsts[-2][0]
        #             local_pos = reduced_nonfuse_lsts[-1][-1]

        #     # always cache read here
        #     if share_pos is not None:
        #         for share in read_cache_share_lst:
        #             s[share].compute_at(s[write_cache], share_pos)
        #     else:
        #         for share in read_cache_share_lst:
        #             s[share].compute_inline()
        #     if local_pos is not None:
        #         for local in read_cache_local_lst:
        #             s[local].compute_at(s[write_cache], local_pos)
        #     else:
        #         for local in read_cache_local_lst:
        #             s[local].compute_inline()
            
        #     # always cooperative fetching
        #     if share_pos is not None:
        #         for share in read_cache_share_lst:
        #             fuse_lst = s[share].op.axis
        #             fused = s[share].fuse(*fuse_lst)
        #             outer, inner = s[share].split(fused, nparts=thread_extents)
        #             s[share].bind(outer, tvm.te.thread_axis("threadIdx.x"))
            
        #     # unroll
        #     if "unroll" in config and len(config["unroll"]) > 0:
        #         step = config["unroll"][0][0]
        #         explicit = config["unroll"][0][1]
        #         s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        #         s[op].pragma(kernel_scope, 'unroll_explicit', explicit)
        # NOTE: COMMENTED OUT BECAUSE IS DEAD CODE (Original FlexTensor had never used it)
        # def _cuda_schedule_fuse_split(s, op, op_state):
        #     # assert_print(op in s)

        #     # always cache write here
        #     # if op.num_outputs > 1:
        #     #     raise RuntimeWarning("Too many outputs in one operation!")
        #     write_cache = s.cache_write(op.output(0), "local")

        #     # always cache read here
        #     read_cache_share_lst = []
        #     # read_cache_local_lst = []
        #     for t in op.input_tensors:
        #         share = s.cache_read(t, "shared", [write_cache])
        #         read_cache_share_lst.append(share)
        #         # local = s.cache_read(share, "local", [write_cache])
        #         # read_cache_local_lst.append(local)
            
        #     # spatial fuse
        #     spatial_axes = s[op].op.axis
        #     fused_spatial_axes = []
        #     if "fuse" in config and len(config["fuse"]) > 0:
        #         # fuse redundant axes
        #         beg = 0
        #         for end in config["fuse"][0]:
        #             fuse_lst = spatial_axes[beg:end]
        #             beg = end
        #             if len(fuse_lst) > 0:
        #                 fused = s[op].fuse(*fuse_lst)
        #                 fused_spatial_axes.append(fused)
        #     else:
        #         fused_spatial_axes = spatial_axes

        #     # spatial split
        #     split_factor_lst = []
        #     splited_spatial_axes = []
        #     if "spatial" in config and len(config["spatial"]) > 0:
        #         # to align each axis
        #         assert len(config["spatial"]) == len(spatial_axes), "align failed"
        #         # compute split factors
        #         if "fuse" in config and len(config["fuse"]) > 0:
        #             beg = 0
        #             for end in config["fuse"][0]:
        #                 tmp_lst = [1] * len(config["spatial"][0])
        #                 for i in range(beg, end):
        #                     for j in range(len(config["spatial"][i])):
        #                         tmp_lst[j] *= config["spatial"][i][j]
        #                 if beg < end:
        #                     split_factor_lst.append(tmp_lst)
        #                 beg = end
        #         else:
        #             split_factor_lst = config["spatial"]
        #         assert len(fused_spatial_axes) == len(split_factor_lst), "align failed"
        #         for axis, nparts in zip(fused_spatial_axes, split_factor_lst):
        #             tmp_buffer = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[op].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #             tmp_buffer.append(axis)
        #             splited_spatial_axes.append(tmp_buffer)
        #     else:
        #         for axis in fused_spatial_axes:
        #             splited_spatial_axes.append([axis])
        #     assert len(splited_spatial_axes) > 0, "empty spatial axes"     # must be non-empty

        #     # always reorder here
        #     reorder_lst = []
        #     for count in range(len(splited_spatial_axes[0])):
        #         tmp_buffer = [x[count] for x in splited_spatial_axes]
        #         reorder_lst.extend(tmp_buffer)
        #     s[op].reorder(*reorder_lst)

        #     # fix kernel scope
        #     kernel_scope = reorder_lst[0]
            
        #     # always bind here
        #     # - prepare thread axis
        #     bx = tvm.te.thread_axis("blockIdx.x")
        #     by = tvm.te.thread_axis("blockIdx.y")
        #     bz = tvm.te.thread_axis("blockIdx.z")
        #     vx = tvm.te.thread_axis("vthread")
        #     vy = tvm.te.thread_axis("vthread")
        #     vz = tvm.te.thread_axis("vthread")
        #     tx = tvm.te.thread_axis("threadIdx.x")
        #     ty = tvm.te.thread_axis("threadIdx.y")
        #     tz = tvm.te.thread_axis("threadIdx.z")

        #     blocks = [bz, by, bx]
        #     threads = [tz, ty, tx]
        #     vthreads = [vz, vy, vx]

        #     block_extents = [-1, -1, -1]    # z, y, x
        #     virtual_extents = [-1, -1, -1]
        #     thread_extents = [-1, -1, -1]

        #     length = len(splited_spatial_axes)
        #     assert length >= 1
        #     # - bind
        #     count = min(length, len(blocks)) - 1
        #     while count >= 0:
        #         parts = len(splited_spatial_axes[count])
        #         assert parts > 0
        #         if parts == 1:
        #             s[op].bind(splited_spatial_axes[count][0], blocks[count])
        #             block_extents[count] = split_factor_lst[count][0]
        #         elif parts == 2:
        #             s[op].bind(splited_spatial_axes[count][0], blocks[count])
        #             block_extents[count] = split_factor_lst[count][0]
        #             s[op].bind(splited_spatial_axes[count][1], threads[count])
        #             thread_extents[count] = split_factor_lst[count][1]
        #         else:
        #             s[op].bind(splited_spatial_axes[count][0], blocks[count])
        #             block_extents[count] = split_factor_lst[count][0]
        #             s[op].bind(splited_spatial_axes[count][1], vthreads[count])
        #             virtual_extents[count] = split_factor_lst[count][1]
        #             s[op].bind(splited_spatial_axes[count][2], threads[count])
        #             thread_extents[count] = split_factor_lst[count][2]
        #         count -= 1
        #     # - compute at pos
        #     count = min(length, len(blocks)) - 1
        #     parts = len(splited_spatial_axes[count])
        #     thread_pos = splited_spatial_axes[count][min(parts - 1, 2)]

        #     # always compute at here
        #     s[write_cache].compute_at(s[op], thread_pos)

        #     # reduce_split
        #     reduced_axes = s[write_cache].op.reduce_axis
        #     splited_reduced_axes = []
        #     if "reduce" in config and len(config["reduce"]) > 0:
        #         # to align each axis
        #         assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        #         for axis, nparts in zip(reduced_axes, config["reduce"]):
        #             tmp_buffer = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[write_cache].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #             tmp_buffer.append(axis)
        #             splited_reduced_axes.append(tmp_buffer)
        #     else:
        #         for axis in reduced_axes:
        #             splited_reduced_axes.append([axis])
        #     share_pos = None
        #     # local_pos = None
        #     # if has reduce axes
        #     if len(splited_reduced_axes) > 0:
        #         # always reorder here
        #         reduced_nonfuse_lsts = []
        #         reorder_lst = []
        #         length = len(splited_reduced_axes[0])
        #         # leave the last part
        #         for count in range(length - 1):
        #             tmp_buffer = [x[count] for x in splited_reduced_axes]
        #             reduced_nonfuse_lsts.append(tmp_buffer)
        #             reorder_lst.extend(tmp_buffer)
        #         # the last part
        #         last_part = [x[length - 1] for x in splited_reduced_axes]
        #         spatial_remainder = s[write_cache].op.axis
        #         # change the order of reduce axes and spatial axes
        #         if "reorder" in config and len(config["reorder"]) > 0:
        #             pos = config["reorder"][0][0]
        #             assert pos < len(spatial_remainder)
        #             tmp_buffer = []
        #             count = len(spatial_remainder) - 1
        #             while count > pos:
        #                 tmp_buffer.append(spatial_remainder[count])
        #                 count -= 1
        #             p = pos
        #             q = len(last_part) - 1
        #             while p >= 0 and q >= 0:
        #                 tmp_buffer.append(spatial_remainder[p])
        #                 tmp_buffer.append(last_part[q])
        #                 p -= 1
        #                 q -= 1
        #             while p >= 0:
        #                 tmp_buffer.append(spatial_remainder[p])
        #                 p -= 1
        #             while q >= 0:
        #                 tmp_buffer.append(last_part[q])
        #                 q -= 1
        #             tmp_buffer = list(reversed(tmp_buffer))
        #             reorder_lst.extend(tmp_buffer)
        #         else:
        #             reorder_lst.extend(last_part)
        #             reorder_lst.extend(spatial_remainder)
        #         s[write_cache].reorder(*reorder_lst)
        #         # decide where to compute at
        #         if length == 1:
        #             share_pos = last_part[-1]
        #         else:
        #             mid = math.ceil(length / 2.0) - 1
        #             share_pos = reduced_nonfuse_lsts[mid][-1]
        #             # local_pos = last_part[-1]

        #     # always cache read here
        #     if share_pos is not None:
        #         for share in read_cache_share_lst:
        #             s[share].compute_at(s[write_cache], share_pos)
        #     else:
        #         for share in read_cache_share_lst:
        #             s[share].compute_inline()
        #     # if local_pos is not None:
        #     #     for local in read_cache_local_lst:
        #     #         s[local].compute_at(s[write_cache], local_pos)
        #     # else:
        #     #     for local in read_cache_local_lst:
        #     #         s[local].compute_inline()
            
        #     # always cooperative fetching
        #     if share_pos is not None:
        #         for share in read_cache_share_lst:
        #             fuse_lst = s[share].op.axis
        #             fused = s[share].fuse(*fuse_lst)
        #             count = 2
        #             cur = 1
        #             limit = 1024
        #             while count >= 0:
        #                 factor = thread_extents[count]
        #                 if factor < 0:
        #                     defined = False
        #                     factor = 16
        #                 else:
        #                     defined = True
        #                 cur *= factor
        #                 if not defined and cur > limit:
        #                     break
        #                 fused, inner = s[share].split(fused, factor=factor)
        #                 s[share].bind(inner, threads[count])
        #                 count -= 1
            
        #     # unroll
        #     if "unroll" in config and len(config["unroll"]) > 0:
        #         step = config["unroll"][0][0]
        #         explicit = config["unroll"][0][1]
        #         s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        #         s[op].pragma(kernel_scope, 'unroll_explicit', explicit)

        def _cuda_schedule_split_reorder_fuse(s, op, op_state):
            loop_lst = []
            loop_idx = []

            # always cache write here
            # if op.num_outputs > 1:
            #     raise RuntimeWarning("Too many outputs in one operation!")
            write_cache = s.cache_write(op.output(0), "local")
            # always cache read here
            read_cache_share_lst = []
            # read_cache_local_lst = []
            for t in op.input_tensors:
                share = s.cache_read(t, "shared", [write_cache])
                read_cache_share_lst.append(share)
                # local = s.cache_read(share, "local", [write_cache])
                # read_cache_local_lst.append(local)

            # spatial split
            spatial_axes = [axis for axis in s[op].op.axis]
            assert len(spatial_axes) > 0, "empty spatial axes"     # must be non-empty
            n = spatial_axes[0]
            kernel_scope, n = s[op].split(n, nparts=1)
            spatial_axes[0] = n
            splited_spatial_axes = []
            splited_spatial_extents = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert len(config["spatial"]) == len(spatial_axes), "align failed"
                for axis, nparts in zip(spatial_axes, config["spatial"]):
                    tmp_buffer = []
                    tmp_extents = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[op].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                        tmp_extents.append(nparts[count])
                    tmp_buffer.append(axis)
                    tmp_extents.append(nparts[-1])
                    splited_spatial_axes.append(tmp_buffer)
                    splited_spatial_extents.append(tmp_extents)
            else:
                for axis in spatial_axes:
                    splited_spatial_axes.append([axis])
                    splited_spatial_extents.append([axis.dom.extent.value])

            # always reorder here
            reorder_lst = []
            reorder_parts = []
            reorder_part_extents = []
            for count in range(len(splited_spatial_axes[0])):
                tmp_buffer = [x[count] for x in splited_spatial_axes]
                tmp_extents = [x[count] for x in splited_spatial_extents]
                reorder_lst.extend(tmp_buffer)
                reorder_parts.append(tmp_buffer)
                reorder_part_extents.append(tmp_extents)
            s[op].reorder(*reorder_lst)
            # handle fuse request
            fused_parts = []
            fused_part_extents = []
            fused_part_idx = []
            if "fuse" in config and len(config["fuse"]) > 0:
                base_id = 0
                for part, extents in zip(reorder_parts, reorder_part_extents):
                    tmp_part = []
                    tmp_extents = []
                    tmp_idx = []
                    idx = 0
                    beg = 0
                    for end in config["fuse"][0]:
                        if end - beg > 1:
                            fuse_lst = part[beg:end]
                            fused = s[op].fuse(*fuse_lst)
                            tmp_part.append(fused)
                            extent = reduce(lambda x, y: x * y, extents[beg:end], 1)
                            tmp_idx.extend([idx] * (end - beg))
                            idx += 1
                            tmp_extents.append(extent)
                        elif end - beg == 1:
                            tmp_part.append(part[beg])
                            tmp_extents.append(extents[beg])
                            tmp_idx.append(idx)
                            idx += 1
                        beg = end
                    fused_parts.append(tmp_part)
                    fused_part_extents.append(tmp_extents)
                    fused_part_idx.append(tmp_idx)

                    loop_lst.extend(tmp_part)
                    loop_idx.extend([x + base_id for x in tmp_idx])
                    base_id += len(tmp_part)
            else:
                fused_parts = reorder_parts
                fused_part_extents = reorder_part_extents
                fused_part_idx = [list(range(len(x))) for x in reorder_parts]

                loop_lst = reorder_lst
                loop_idx = list(range(len(reorder_lst)))
            # record op state
            op_state.loop_lst = loop_lst
            op_state.loop_idx = loop_idx
      
            # always bind here
            # - prepare thread axis
            bx = tvm.te.thread_axis("blockIdx.x")
            by = tvm.te.thread_axis("blockIdx.y")
            bz = tvm.te.thread_axis("blockIdx.z")
            vx = tvm.te.thread_axis("vthread")
            vy = tvm.te.thread_axis("vthread")
            vz = tvm.te.thread_axis("vthread")
            tx = tvm.te.thread_axis("threadIdx.x")
            ty = tvm.te.thread_axis("threadIdx.y")
            tz = tvm.te.thread_axis("threadIdx.z")

            blocks = [bz, by, bx]
            threads = [tz, ty, tx]
            vthreads = [vz, vy, vx]

            block_extents = [-1, -1, -1]    # z, y, x
            virtual_extents = [-1, -1, -1]
            thread_extents = [-1, -1, -1]

            bind_option = [None, None, None]
            bind_candidate = [blocks, vthreads, threads]
            candiate_extents = [block_extents, virtual_extents, thread_extents]

            # - bind
            num_parts = len(fused_parts)
            if num_parts == 1:
                bind_option[0] = (fused_parts[0], fused_part_extents[0])
                local_pos = fused_parts[0][:len(bind_candidate[0])][-1]
            elif num_parts == 2:
                bind_option[0] = (fused_parts[0], fused_part_extents[0])
                bind_option[2] = (fused_parts[1], fused_part_extents[1])
                local_pos = fused_parts[1][:len(bind_candidate[2])][-1]
            else:
                bind_option[0] = (fused_parts[0], fused_part_extents[0])
                bind_option[1] = (fused_parts[1], fused_part_extents[1])
                bind_option[2] = (fused_parts[2], fused_part_extents[2])
                local_pos = fused_parts[2][:len(bind_candidate[2])][-1]
            for option, candidate, extents in zip(bind_option, bind_candidate, candiate_extents):
                if option is not None:
                    for i, axis in enumerate(option[0][:len(candidate)]):
                        s[op].bind(axis, candidate[i])
                        extents[i] = option[1][i]
            # compute at
            if "local_pos" in config and len(config["local_pos"]) > 0:
                local_at_part = config["local_pos"][0][0]
                local_at_idx = config["local_pos"][0][1]
                # index changed because of fusion
                cur_idx = fused_part_idx[local_at_part][local_at_idx]
                local_pos = fused_parts[local_at_part][cur_idx]

            # always compute at here
            s[write_cache].compute_at(s[op], local_pos)

            # reduce_split
            reduced_axes = s[write_cache].op.reduce_axis
            splited_reduced_axes = []
            if "reduce" in config and len(config["reduce"]) > 0:
                # to align each axis
                assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
                for axis, nparts in zip(reduced_axes, config["reduce"]):
                    tmp_buffer = []
                    for count in range(len(nparts) - 1):
                        outer, axis = s[write_cache].split(axis, nparts=nparts[count])
                        tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_reduced_axes.append(tmp_buffer)
            else:
                for axis in reduced_axes:
                    splited_reduced_axes.append([axis])
            share_pos = None
            # local_pos = None
            # if has reduce axes
            if len(splited_reduced_axes) > 0:
                # always reorder here
                reduced_nonfuse_lsts = []
                reorder_lst = []
                length = len(splited_reduced_axes[0])
                # leave the last part
                for count in range(length - 1):
                    tmp_buffer = [x[count] for x in splited_reduced_axes]
                    reduced_nonfuse_lsts.append(tmp_buffer)
                    reorder_lst.extend(tmp_buffer)
                # the last part
                last_part = [x[length - 1] for x in splited_reduced_axes]
                spatial_remainder = s[write_cache].op.axis
                # change the order of reduce axes and spatial axes
                if "reorder" in config and len(config["reorder"]) > 0:
                    pos = config["reorder"][0][0]
                    assert pos < len(spatial_remainder)
                    tmp_buffer = []
                    count = len(spatial_remainder) - 1
                    while count > pos:
                        tmp_buffer.append(spatial_remainder[count])
                        count -= 1
                    p = pos
                    q = len(last_part) - 1
                    while p >= 0 and q >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        tmp_buffer.append(last_part[q])
                        p -= 1
                        q -= 1
                    while p >= 0:
                        tmp_buffer.append(spatial_remainder[p])
                        p -= 1
                    while q >= 0:
                        tmp_buffer.append(last_part[q])
                        q -= 1
                    tmp_buffer = list(reversed(tmp_buffer))
                    reorder_lst.extend(tmp_buffer)
                else:
                    reorder_lst.extend(last_part)
                    reorder_lst.extend(spatial_remainder)
                s[write_cache].reorder(*reorder_lst)
                # decide where to compute at
                if "share_pos" in config and len(config["share_pos"]) > 0:
                    share_at = config["share_pos"][0][0]
                    share_idx = config["share_pos"][0][1]
                    reduced_nonfuse_lsts.append(last_part)
                    share_pos = reduced_nonfuse_lsts[share_at][share_idx]
                else:
                    if length == 1:
                        share_pos = last_part[-1]
                    else:
                        mid = math.ceil(length / 2.0) - 1
                        share_pos = reduced_nonfuse_lsts[mid][-1]
                        # local_pos = last_part[-1]

            # always cache read here
            if share_pos is not None:
                for share in read_cache_share_lst:
                    s[share].compute_at(s[write_cache], share_pos)
            else:
                for share in read_cache_share_lst:
                    s[share].compute_inline()
            # if local_pos is not None:
            #     for local in read_cache_local_lst:
            #         s[local].compute_at(s[write_cache], local_pos)
            # else:
            #     for local in read_cache_local_lst:
            #         s[local].compute_inline()
            
            # always cooperative fetching
            if share_pos is not None:
                for share in read_cache_share_lst:
                    fuse_lst = s[share].op.axis
                    fused = s[share].fuse(*fuse_lst)
                    count = 2
                    cur = 1
                    limit = 1024
                    while count >= 0:
                        factor = thread_extents[count]
                        if factor < 0:
                            defined = False
                            factor = 16
                        else:
                            defined = True
                        cur *= factor
                        if not defined and cur > limit:
                            break
                        fused, inner = s[share].split(fused, factor=factor)
                        s[share].bind(inner, threads[count])
                        count -= 1
            
            # unroll
            if "unroll" in config and len(config["unroll"]) > 0:
                step = config["unroll"][0][0]
                explicit = config["unroll"][0][1]
                s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
                s[op].pragma(kernel_scope, 'unroll_explicit', explicit)

        # NOTE: COMMENTED OUT BECAUSE IS DEAD CODE (Original FlexTensor had never used it)
        # def _cpu_schedule_split_fuse(s, op, op_state):
        #     # always cache write here
        #     # if op.num_outputs > 1:
        #     #     raise RuntimeWarning("Too many outputs in one operation!")
        #     write_cache = s.cache_write(op.output(0), "global")

        #     # spatial split
        #     spatial_axes = s[op].op.axis
        #     splited_spatial_axes = []
        #     if "spatial" in config and len(config["spatial"]) > 0:
        #         # to align each axis
        #         assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
        #         for axis, nparts in zip(spatial_axes, config["spatial"]):
        #             tmp_buffer = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[op].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #             tmp_buffer.append(axis)
        #             splited_spatial_axes.append(tmp_buffer)
        #     else:
        #         for axis in spatial_axes:
        #             splited_spatial_axes.append([axis])
        #     assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")     # must be non-empty

        #     # always reorder and fuse here
        #     spatial_fuse_lsts = []
        #     spatial_fuse_extents = []
        #     reorder_lst = []
        #     fused_spatial_axes = []
        #     for count in range(len(splited_spatial_axes[0])):
        #         tmp_buffer = [x[count] for x in splited_spatial_axes]
        #         tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
        #         spatial_fuse_lsts.append(tmp_buffer)
        #         spatial_fuse_extents.append(tmp_extent)
        #         reorder_lst.extend(tmp_buffer)
        #     s[op].reorder(*reorder_lst)
        #     for fuse_lst in spatial_fuse_lsts:
        #         fused = s[op].fuse(*fuse_lst)
        #         fused_spatial_axes.append(fused)
        #     kernel_scope = fused_spatial_axes[0]
            
        #     # always parallel here
        #     length = len(fused_spatial_axes)
        #     assert_print(length > 0, "empty spatial axes!")
        #     s[op].parallel(fused_spatial_axes[0])
        #     if length == 1:
        #         thread_pos = fused_spatial_axes[0]
        #     if 2 <= length <= 3:
        #         thread_pos = fused_spatial_axes[1]
        #     else:
        #         thread_pos = fused_spatial_axes[2]

        #     # always compute at here
        #     s[write_cache].compute_at(s[op], thread_pos)

        #     # reduce_split
        #     reduced_axes = s[write_cache].op.reduce_axis
        #     splited_reduced_axes = []
        #     if "reduce" in config and len(config["reduce"]) > 0:
        #         # to align each axis
        #         assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        #         for axis, nparts in zip(reduced_axes, config["reduce"]):
        #             tmp_buffer = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[write_cache].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #             tmp_buffer.append(axis)
        #             splited_reduced_axes.append(tmp_buffer)
        #     else:
        #         for axis in reduced_axes:
        #             splited_reduced_axes.append([axis])

        #     # if has reduce axes
        #     if len(splited_reduced_axes) > 0:
        #         # always reorder here
        #         reduced_nonfuse_lsts = []
        #         reorder_lst = []
        #         length = len(splited_reduced_axes[0])
                
        #         for count in range(length):
        #             tmp_buffer = [x[count] for x in splited_reduced_axes]
        #             reduced_nonfuse_lsts.append(tmp_buffer)
        #             reorder_lst.extend(tmp_buffer)
        #         # change the order of reduce axes and spatial axes
        #         rlength = len(splited_reduced_axes)
        #         if rlength > 1:
        #             reorder_lst.extend(s[write_cache].op.axis)
        #         elif rlength == 1:   # in this case, have to interleave otherwise the reorder is of no use
        #             tmp_order = []
        #             p_spatial = len(s[write_cache].op.axis) - 1
        #             p_reduce = len(reorder_lst) - 1
        #             while p_spatial >= 0 and p_reduce >= 0:
        #                 tmp_order.append(s[write_cache].op.axis[p_spatial])
        #                 tmp_order.append(reorder_lst[p_reduce])
        #                 p_spatial -= 1
        #                 p_reduce -= 1
        #             while p_spatial >= 0:
        #                 tmp_order.append(s[write_cache].op.axis[p_spatial])
        #                 p_spatial -= 1
        #             while p_reduce >= 0:
        #                 tmp_order.append(reorder_lst[p_reduce])
        #                 p_reduce -= 1
        #             tmp_order = list(reversed(tmp_order))
        #             reorder_lst = tmp_order
        #         s[write_cache].reorder(*reorder_lst)
            
        #     # unroll
        #     if "unroll" in config and len(config["unroll"]) > 0:
        #         step = config["unroll"][0][0]
        #         s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
            
        #     # always vectorize here
        #     s[write_cache].vectorize(s[write_cache].op.axis[-1])

        # NOTE: COMMENTED OUT BECAUSE IS DEAD CODE (Original FlexTensor had never used it)
        # def _cpu_schedule_split_reorder_fuse(s, op, op_state):
        #     # assert_print(op in s)

        #     loop_idx = []
        #     loop_lst = []

        #     # always cache write here
        #     # if op.num_outputs > 1:
        #     #     raise RuntimeWarning("Too many outputs in one operation!")
        #     write_cache = s.cache_write(op.output(0), "local")

        #     # spatial split
        #     spatial_axes = [axis for axis in s[op].op.axis]
        #     assert len(spatial_axes) > 0, "empty spatial axes"     # must be non-empty
        #     n = spatial_axes[0]
        #     kernel_scope, n = s[op].split(n, nparts=1)
        #     spatial_axes[0] = n

        #     splited_spatial_axes = []
        #     splited_spatial_extents = []
        #     if "spatial" in config and len(config["spatial"]) > 0:
        #         # to align each axis
        #         assert len(config["spatial"]) == len(spatial_axes), "align failed"
        #         for axis, nparts in zip(spatial_axes, config["spatial"]):
        #             tmp_buffer = []
        #             tmp_extents = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[op].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #                 tmp_extents.append(nparts[count])
        #             tmp_buffer.append(axis)
        #             tmp_extents.append(nparts[-1])
        #             splited_spatial_axes.append(tmp_buffer)
        #             splited_spatial_extents.append(tmp_extents)
        #     else:
        #         for axis in spatial_axes:
        #             splited_spatial_axes.append([axis])
        #             splited_spatial_extents.append([axis.dom.extent.value])

        #     # always reorder here
        #     reorder_lst = []
        #     reorder_parts = []
        #     reorder_part_extents = []
        #     for count in range(len(splited_spatial_axes[0])):
        #         tmp_buffer = [x[count] for x in splited_spatial_axes]
        #         tmp_extents = [x[count] for x in splited_spatial_extents]
        #         reorder_lst.extend(tmp_buffer)
        #         reorder_parts.append(tmp_buffer)
        #         reorder_part_extents.append(tmp_extents)
        #     s[op].reorder(*reorder_lst)

        #     # handle fuse request
        #     fused_parts = []
        #     fused_part_extents = []
        #     fused_part_idx = []
        #     if "fuse" in config and len(config["fuse"]) > 0:
        #         base_id = 0
        #         for part, extents in zip(reorder_parts, reorder_part_extents):
        #             tmp_part = []
        #             tmp_extents = []
        #             tmp_idx = []
        #             idx = 0
        #             beg = 0
        #             for end in config["fuse"][0]:
        #                 if end - beg > 1:
        #                     fuse_lst = part[beg:end]
        #                     fused = s[op].fuse(*fuse_lst)
        #                     tmp_part.append(fused)
        #                     extent = reduce(lambda x, y: x * y, extents[beg:end], 1)
        #                     tmp_idx.extend([idx] * (end - beg))
        #                     idx += 1
        #                     tmp_extents.append(extent)
        #                 elif end - beg == 1:
        #                     tmp_part.append(part[beg])
        #                     tmp_extents.append(extents[beg])
        #                     tmp_idx.append(idx)
        #                     idx += 1
        #                 beg = end
        #             fused_parts.append(tmp_part)
        #             fused_part_extents.append(tmp_extents)
        #             fused_part_idx.append(tmp_idx)

        #             # for op state
        #             loop_lst.extend(tmp_part)
        #             loop_idx.extend([x + base_id for x in tmp_idx])
        #             base_id += len(tmp_part)
        #     else:
        #         fused_parts = reorder_parts
        #         fused_part_extents = reorder_part_extents
        #         fused_part_idx = [list(range(len(x))) for x in reorder_parts]

        #         # for op state
        #         loop_lst = reorder_lst
        #         loop_idx = list(range(len(reorder_lst)))

        #     # record op state
        #     op_state.loop_lst = loop_lst
        #     op_state.loop_idx = loop_idx

        #     # parallel
        #     fused = s[op].fuse(*fused_parts[0])
        #     s[op].parallel(fused)
     
        #     # compute at
        #     num_parts = len(fused_parts)
        #     if num_parts == 1:
        #         local_pos = fused
        #     elif num_parts == 2:
        #         local_pos = fused_parts[num_parts-1][0]
        #     else:
        #         local_pos = fused_parts[num_parts-2][-1]

        #     if "local_pos" in config and len(config["local_pos"]) > 0:
        #         local_at_part = config["local_pos"][0][0]
        #         local_at_idx = config["local_pos"][0][1]
        #         # index changed because of fusion
        #         cur_idx = fused_part_idx[local_at_part][local_at_idx]
        #         local_pos = fused_parts[local_at_part][cur_idx]

        #     # always compute at here
        #     s[write_cache].compute_at(s[op], local_pos)

        #     # reduce_split
        #     reduced_axes = s[write_cache].op.reduce_axis
        #     splited_reduced_axes = []
        #     if "reduce" in config and len(config["reduce"]) > 0:
        #         # to align each axis
        #         assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
        #         for axis, nparts in zip(reduced_axes, config["reduce"]):
        #             tmp_buffer = []
        #             for count in range(len(nparts) - 1):
        #                 outer, axis = s[write_cache].split(axis, nparts=nparts[count])
        #                 tmp_buffer.append(outer)
        #             tmp_buffer.append(axis)
        #             splited_reduced_axes.append(tmp_buffer)
        #     else:
        #         for axis in reduced_axes:
        #             splited_reduced_axes.append([axis])

        #     # if has reduce axes
        #     if len(splited_reduced_axes) > 0:
        #         # always reorder here
        #         reduced_nonfuse_lsts = []
        #         reorder_lst = []
        #         length = len(splited_reduced_axes[0])
        #         # leave the last part
        #         for count in range(length - 1):
        #             tmp_buffer = [x[count] for x in splited_reduced_axes]
        #             reduced_nonfuse_lsts.append(tmp_buffer)
        #             reorder_lst.extend(tmp_buffer)
        #         # the last part
        #         last_part = [x[length - 1] for x in splited_reduced_axes]
        #         spatial_remainder = s[write_cache].op.axis
        #         # change the order of reduce axes and spatial axes
        #         if "reorder" in config and len(config["reorder"]) > 0:
        #             pos = config["reorder"][0][0]
        #             assert pos < len(spatial_remainder)
        #             tmp_buffer = []
        #             count = len(spatial_remainder) - 1
        #             while count > pos:
        #                 tmp_buffer.append(spatial_remainder[count])
        #                 count -= 1
        #             p = pos
        #             q = len(last_part) - 1
        #             while p >= 0 and q >= 0:
        #                 tmp_buffer.append(spatial_remainder[p])
        #                 tmp_buffer.append(last_part[q])
        #                 p -= 1
        #                 q -= 1
        #             while p >= 0:
        #                 tmp_buffer.append(spatial_remainder[p])
        #                 p -= 1
        #             while q >= 0:
        #                 tmp_buffer.append(last_part[q])
        #                 q -= 1
        #             tmp_buffer = list(reversed(tmp_buffer))
        #             reorder_lst.extend(tmp_buffer)
        #         else:
        #             reorder_lst.extend(last_part)
        #             reorder_lst.extend(spatial_remainder)
        #         s[write_cache].reorder(*reorder_lst)
            
        #     # unroll
        #     if "unroll" in config and len(config["unroll"]) > 0:
        #         step = config["unroll"][0][0]
        #         explicit = config["unroll"][0][1]
        #         s[op].pragma(kernel_scope, 'auto_unroll_max_step', step)
        #         s[op].pragma(kernel_scope, 'unroll_explicit', explicit)

        def _cpu_schedule_simple(s, op, op_state):
            # always cache write here
            # if op.num_outputs > 1:
            #     raise RuntimeWarning("Too many outputs in one operation!")
            write_cache = s.cache_write(op.output(0), "global")
            # spatial split
            spatial_axes = s[op].op.axis
            splited_spatial_axes = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
                for axis, nparts in zip(spatial_axes, config["spatial"]):
                    nfactors = [1]
                    count = len(nparts) - 1
                    while count >= 0:
                        nfactors.append(nparts[count] * nfactors[-1])
                        count -= 1
                    tmp_buffer = []
                    num_factors = len(nfactors)
                    for i in range(num_factors - 2):
                        factor = nfactors[num_factors - 2 - i]
                        part = nparts[i]
                        if factor == 1:
                            tmp_buffer.append(axis)
                            axis = None
                        elif part == 1:
                            tmp_buffer.append(None)
                        else:
                            outer, axis = s[op].split(axis, factor=factor)
                            tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_spatial_axes.append(tmp_buffer)
            else:
                for axis in spatial_axes:
                    splited_spatial_axes.append([axis])
            assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")     # must be non-empty

            # always reorder and fuse here
            # this part actually suppose there is "spatial" in config
            # which is avoidable
            spatial_fuse_lsts = []
            spatial_fuse_extents = []
            reorder_lst = []
            fused_spatial_axes = []
            spatial_split_num_parts = len(splited_spatial_axes[0])
            for count in range(spatial_split_num_parts):
                tmp_buffer = [x[count] for x in splited_spatial_axes]
                tmp_extent = reduce(lambda a, b: a * b, [x[count] for x in config["spatial"]])
                spatial_fuse_lsts.append(tmp_buffer)
                spatial_fuse_extents.append(tmp_extent)
                reorder_lst.extend(tmp_buffer)
            reorder_lst_without_none = list(filter(lambda x: x is not None, reorder_lst))
            s[op].reorder(*reorder_lst_without_none)
            for fuse_lst in spatial_fuse_lsts[:1]:
                tmp_buffer = list(filter(lambda x: x is not None, fuse_lst))
                fused = s[op].fuse(*tmp_buffer)
                fused_spatial_axes.append(fused)
            kernel_scope = fused_spatial_axes[0]
            if len(spatial_fuse_lsts) > 1:
                count = 0
                while count < len(config["spatial"]) and config["spatial"][count][1] == 1:
                    count += 1
                if count == len(config["spatial"]):
                    count -= 1
                next_pos_for_comptue_at = spatial_fuse_lsts[1][count]
            else:
                next_pos_for_comptue_at = kernel_scope 
            
            # always parallel here
            s[op].parallel(kernel_scope)
            # vectorize
            if len(spatial_fuse_lsts) == 2:
                count = len(spatial_fuse_lsts[1]) - 1
                while count >= 1:
                    if spatial_fuse_lsts[1][count] is not None and config["spatial"][1][count] > 1:
                        s[op].vectorize(spatial_fuse_lsts[1][count])
                        break
                    count -= 1
            elif len(spatial_fuse_lsts) > 2:
                count = len(spatial_fuse_lsts[-1]) - 1
                while count >= 0:
                    if spatial_fuse_lsts[-1][count] is not None and config["spatial"][count][-1] > 1:
                        s[op].vectorize(spatial_fuse_lsts[-1][count])
                        break
                    count -= 1
            # always compute at here
            s[write_cache].compute_at(s[op], next_pos_for_comptue_at)

            # spatial_split for write cache
            spatial_axes = s[write_cache].op.axis
            num_spatial_axes = len(spatial_axes)
            splited_spatial_axes = []
            if "spatial" in config and len(config["spatial"]) > 0:
                # to align each axis
                assert_print(len(config["spatial"]) == len(spatial_axes), "align failed")
                for axis, nparts in zip(spatial_axes, config["spatial"]):
                    nfactors = [1]
                    count = len(nparts) - 1
                    while count >= 0:
                        nfactors.append(nparts[count] * nfactors[-1])
                        count -= 1
                    tmp_buffer = []
                    num_factors = len(nfactors)
                    for i in range(num_factors - 2):
                        factor = nfactors[num_factors - 2 - i]
                        part = nparts[i]
                        if factor == 1:
                            tmp_buffer.append(axis)
                            axis = None
                        elif part == 1:
                            tmp_buffer.append(None)
                        else:
                            outer, axis = s[write_cache].split(axis, factor=factor)
                            tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_spatial_axes.append(tmp_buffer)
            else:
                for axis in spatial_axes:
                    splited_spatial_axes.append([axis])
            assert_print(len(splited_spatial_axes) > 0, "empty spatial axes")     # must be non-empty
            # reduce_split for write cache
            reduced_axes = s[write_cache].op.reduce_axis
            num_reduce_axes = len(reduced_axes)
            splited_reduced_axes = []
            if "reduce" in config and len(config["reduce"]) > 0:
                # to align each axis
                assert_print(len(config["reduce"]) == len(reduced_axes), "align reduce failed")
                for axis, nparts in zip(reduced_axes, config["reduce"]):
                    nfactors = [1]
                    count = len(nparts) - 1
                    while count >= 0:
                        nfactors.append(nparts[count] * nfactors[-1])
                        count -= 1
                    tmp_buffer = []
                    num_factors = len(nfactors)
                    for i in range(num_factors - 2):
                        factor = nfactors[num_factors - 2 - i]
                        part = nparts[i]
                        if factor == 1:
                            tmp_buffer.append(axis)
                            axis = None
                        elif part == 1:
                            tmp_buffer.append(None)
                        else:
                            outer, axis = s[write_cache].split(axis, factor=factor)
                            tmp_buffer.append(outer)
                    tmp_buffer.append(axis)
                    splited_reduced_axes.append(tmp_buffer)
            else:
                for axis in reduced_axes:
                    splited_reduced_axes.append([axis])
            # for easy align
            # reduce_split_num_parts = len(splited_reduced_axes[0])
            # assert reduce_split_num_parts == spatial_split_num_parts
            # reorder hybrid for spatial and reduce
            hybrid_axes = splited_spatial_axes + splited_reduced_axes
            hybrid_fuse_lsts = []
            hybrid_reorder_lst = []
            for count in range(spatial_split_num_parts):
                tmp_buffer = [x[count] for x in hybrid_axes]
                hybrid_fuse_lsts.append(tmp_buffer)
                hybrid_reorder_lst.extend(tmp_buffer)
            if len(hybrid_fuse_lsts) > 1:
                last_parts = hybrid_reorder_lst[-num_spatial_axes-num_reduce_axes:]
                hybrid_reorder_lst = hybrid_reorder_lst[:-num_spatial_axes-num_reduce_axes]
                tmp_buffer = last_parts[-num_reduce_axes:]
                tmp_buffer.extend(last_parts[:-num_reduce_axes])
                hybrid_reorder_lst.extend(tmp_buffer)
            hybrid_reorder_lst_without_none = list(filter(lambda x: x is not None, hybrid_reorder_lst))
            s[write_cache].reorder(*hybrid_reorder_lst_without_none)
            # fuse without reduce axes
            # assert len(hybrid_fuse_lsts) > 0
            # s[write_cache].fuse(*hybrid_fuse_lsts[0][:-num_reduce_axes])
            
            # unroll and vectorize without reduce axes
            if len(hybrid_fuse_lsts) > 1:
                rcount = num_spatial_axes - 1
                while rcount >= 0 and config["spatial"][rcount][-1] == 1:
                    rcount -= 1
                if rcount >= 0:
                    s[write_cache].vectorize(hybrid_fuse_lsts[-1][rcount])
                for count in range(rcount):
                    if config["spatial"][count][-1] > 1:
                        s[write_cache].unroll(hybrid_fuse_lsts[-1][count])
            if len(hybrid_fuse_lsts) > 2:
                for count in range(num_spatial_axes):
                    if config["spatial"][count][-2] > 1:
                        s[write_cache].unroll(hybrid_fuse_lsts[-2][count])
                # for count in range(num_reduce_axes):
                #     if config["reduce"][count][-2] > 1:
                #         s[write_cache].unroll(hybrid_fuse_lsts[-2][count + num_spatial_axes])

        if "cuda" in target.keys:
            return _cuda_schedule_split_reorder_fuse
        elif "cpu" in target.keys:
            return _cpu_schedule_simple
        else:
            raise RuntimeError("Currently no support for target %s"%target)  

class GraphScheduler(Scheduler):
    def __init__(self, task, space, perf_path=None, use_model=False, rewrite=False):
        Scheduler.__init__(self, "graph", task, space, perf_path=perf_path, use_model=use_model, rewrite=rewrite)
    
    @staticmethod
    def generate_graph_schedule(config, phase="inline"):
        def _inline_schedule(s, op_lst, op_states):
            if "inline" in config and len(config["inline"]) > 0:
                entity = config["inline"][0]
                for count in range(len(op_lst)):
                    if entity[count]:
                        s[op_lst[count]].compute_inline()
                        op_states[count].inline = True

        def _at_schedule(s, op_lst, op_states):
            return
            # NOTE: Really odd but that's how original flex tensor kept it
            if "merge" in config and len(config["merge"]) > 0:
                entity = config["merge"][0]
                for count in range(len(op_lst)):
                    if entity[count] >= 0:
                        num_consumers = len(op_states[count].consumer_lst)
                        if num_consumers != 1 or op_states[count].inline:
                            continue
                        else:
                            consumer_id = op_states[count].consumer_lst[0]
                            consumer_state = op_states[consumer_id]
                            if consumer_state.inline:
                                continue    # do not compute at inlined ops
                            consumer_loop_idx = consumer_state.loop_idx
                            at_pos = consumer_state.loop_lst[consumer_loop_idx[entity[count]]]
                            s[op_lst[count]].compute_at(s[op_lst[consumer_id]], at_pos)
                            op_states[count].compute_at = True

        if phase == "inline":
            return _inline_schedule
        elif phase == "at":
            return _at_schedule
        else:
            raise RuntimeError("Currently no support for phase %s" %phase)


class OpState(object):
    def __init__(self):
        self.inline = False
        self.loop_lst = []
        self.loop_idx = []
        self.compute_at = False
        self.consumer_lst = []

def schedule_with_config(task, configs, op_pos=None, rewrite=False):
    """Schedule a task with given configs

    perform sequential schedule
    """
    rewriter = Rewriter(configs)
    if rewrite:
        ops, bufs, new_graph_config, new_op_config_lst = rewriter.rewrite(task)
        configs = Config(new_op_config_lst, new_graph_config)
    else:
        func = task.func
        args = task.args
        ops, bufs = func(*args)
    s, buffers = schedule_with_config_ops(ops, bufs, configs, op_pos=op_pos, target=task.target)
    return s, buffers

def schedule_with_config_ops(ops, bufs, configs, op_pos=None, target=None):
    """Schedule a task with given configs

    perform sequential schedule
    """
    # sort the ops, so that we can distinguish each op
    op_lst, down_graph = flatten_graph(ops)

    # state of ops
    op_states = [OpState() for op in op_lst]
    for count_op, op in enumerate(op_lst):
        consumer_lst = []
        for count_output in range(op.num_outputs):
            if op.output(count_output) in down_graph:
                consumer_lst.extend(down_graph[op.output(count_output)])
        op_states[count_op].consumer_lst = list(set(consumer_lst))

    op_config_lst = configs.op_config_lst

    if op_pos is not None:
        assert_print(isinstance(op_pos, int), "op_pos should be int")
        assert_print(op_pos < len(op_lst) and op_pos < len(op_config_lst), "op_pos too big")
        loop_length = op_pos + 1
        s = tvm.te.create_schedule(op_lst[op_pos])
    else:
        assert_print(len(op_config_lst) <= len(op_lst), "config length exceed op_lst")
        loop_length = len(op_config_lst)
        s = tvm.te.create_schedule(ops)

    ###################################################
    # perform inter operations schedule first for inline
    graph_config = configs.graph_config
    if graph_config is not None:
        graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="inline")
        graph_template(s, op_lst, op_states)

    ###################################################
    # perform intra operations schedule    
    for i in range(loop_length):
        # mask inlined ops
        if not op_states[i].inline:
            op = op_lst[i]
            config = op_config_lst[i]
            template = OpScheduler.generate_op_schedule(target, config)
            template(s, op, op_states[i])   

    ###################################################
    # perform inter operations schedule again for compute at
    if graph_config is not None:
        graph_template = GraphScheduler.generate_graph_schedule(graph_config, phase="at")
        graph_template(s, op_lst, op_states)

    return s, bufs