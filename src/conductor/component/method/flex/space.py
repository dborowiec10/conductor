import tvm
import math
import numpy as np
from conductor.component.method.flex.utils import (assert_print, gen_enum, any_factor_split, gen_group, is_power_of_x)

def able_inline(op, down_graph):
    is_compute = isinstance(op, tvm.te.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    is_output = False
    for i in range(op.num_outputs):
        if op.output(i) not in down_graph:
            is_output = True
            break
    return is_compute and (not has_reduce) and (not is_output)

class Space(object):
    def __init__(self):
        self.subspaces = {}
        self.types = {}
        self.valid_type_keys = ["fuse", "spatial", "reduce", "reorder", "inline", "unroll", "merge", "special"]
        for type_key in self.valid_type_keys:
            self.types[type_key] = []
        self.dim = 0

    def __repr__(self):
        subsp = ""
        for k, s in self.subspaces.items():
            subsp += "\n" + str(k) + "-->" + str(s)
        return f"Space(\t{subsp}\n)"

    def add_subspace(self, name, subspace, type_key, override=False):
        if name in self.subspaces and not override:
            raise RuntimeError("Same subspace name")
        assert_print(type_key in self.valid_type_keys)
        self.subspaces[name] = subspace
        self.types[type_key].append(name)
        self.dim += subspace.dim

    def items(self):
        return self.subspaces.items()

    def __len__(self):
        ret = 1
        for _, subspace in self.subspaces.items():
            ret *= len(subspace)
        return ret

    def length(self):
        ret = {}
        total = 1
        added = 0
        for name, subspace in self.subspaces.items():
            ret[name] = len(subspace)
            total *= ret[name]
            added += ret[name]
        ret["total"] = total
        ret["added"] = added
        return ret


DirectedSubSpaceTypeKeys = ["spatial", "reduce"]
UndirectedSubSpaceTypeKeys = ["fuse", "reorder", "unroll", "inline", "merge", "special"]

class SubSpace(object):
    def __init__(self):
        self.dim = 0
        self.static_entities = []
        self.size = 0
        self.num_direction = 0

    def random_entity(self):
        return np.random.choice(self.static_entities)

    def next_entity(self, *args, **kwargs):
        raise NotImplementedError()

    def get_entity(self, p):
        return self.static_entities[p]

    def get_direction(self, num):
        raise NotImplementedError()

    def __len__(self):
        return self.size


class SplitSpace(SubSpace):
    def __init__(self, dim, total, allow_non_divisible='off'):
        super(SplitSpace, self).__init__()
        self.total = total
        self.allow_non_divisible = allow_non_divisible
        self.dim = dim
        self.static_entities = any_factor_split(total, dim, allow_non_divisible=allow_non_divisible)
        self.size = len(self.static_entities)
        self.num_direction = dim * (dim - 1)
        self.directions = []
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    self.directions.append((i, j))
        self.type_key = "split"

    def __repr__(self):
        return f"SplitSpace(static_entities={self.static_entities}, allow_non_divisible={self.allow_non_divisible}, total={self.total}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            next_pos = (pos + d[0]) % self.size
            return next_pos
        elif len(d) == 2:
            asc_pos, dec_pos = d[0], d[1]
            assert_print(0 <= asc_pos < self.dim)
            assert_print(0 <= dec_pos < self.dim)
            assert_print(asc_pos != dec_pos)
            current = self.static_entities[pos]
            ret = current.copy()
            left = current[asc_pos] * current[dec_pos]
            canout = False
            next_pos = -1
            while not canout:
                tmp = ret[asc_pos] + 1
                while tmp <= left:
                    if self.allow_non_divisible == 'continuous':
                        break
                    elif self.allow_non_divisible == 'power2' and is_power_of_x(2, tmp):
                        break
                    elif left % tmp == 0:
                        break
                    tmp += 1
                tmp = min(tmp, left)
                ret[asc_pos] = tmp
                ret[dec_pos] = math.ceil(left / tmp)
                try:
                    next_pos = self.static_entities.index(ret)
                    canout = True
                except ValueError:
                    canout = False
            return next_pos
        else:
            raise RuntimeError(
                "Not support for direction more than two dims: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class FuseSpace(SubSpace):
    def __init__(self, dim, elements):
        self.dim = dim
        self.static_entities = gen_group(elements, most_groups=self.dim)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "fuse"

    def __repr__(self):
        return f"FuseSpace(static_entities={self.static_entities}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    
class ReorderSpace(SubSpace):
    def __init__(self, num_spatial_axis):
        self.dim = 1
        self.static_entities = [[i] for i in range(num_spatial_axis)]
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "reorder"

    def __repr__(self):
        return f"ReorderSpace(static_entities={self.static_entities}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class UnrollSpace(SubSpace):
    def __init__(self, steps, explicit=False):
        super(UnrollSpace, self).__init__()
        self.dim = 2
        self.static_entities = []
        self.steps = steps
        explicits = [1] if explicit else [0, 1]
        for step in steps:
            for _explicit in explicits:
                self.static_entities.append([step, _explicit])
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "unroll"

    def __repr__(self):
        return f"UnrollSpace(static_entities={self.static_entities}, steps={self.steps}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class PosSpace(SubSpace):
    def __init__(self, parts, num_axis):
        self.dim = 2
        self.static_entities = []
        self.parts = parts
        self.num_axis = num_axis
        for i in range(parts):
            for j in range(num_axis):
                self.static_entities.append([i, j])
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "local"

    def __repr__(self):
        return f"PosSpace(static_entities={self.static_entities}, num_axis={self.num_axis}, parts={self.parts}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class InlineSpace(SubSpace):
    def __init__(self, inline_op_pos, op_num, force_inline=False):
        self.dim = op_num
        self.static_entities = []
        self.able_inline_list = inline_op_pos
        if force_inline:
            entity = [0] * op_num
            for pos in inline_op_pos:
                entity[pos] = 1
            self.static_entities.append(entity)
        else:
            num_inline_ops = len(inline_op_pos)
            enums = gen_enum([1, 0], num_inline_ops)
            for enum in enums:
                entity = [0] * op_num
                for i in range(num_inline_ops):
                    entity[inline_op_pos[i]] = enum[i]
                self.static_entities.append(entity)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "inline"

    def __repr__(self):
        return f"InlineSpace(static_entities={self.static_entities}, able_inline_list={self.able_inline_list}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    def able_inline(self, pos):
        return pos in self.able_inline_list


class MergeSpce(SubSpace):
    def __init__(self, merge_op_pos, op_num, force_merge=False):
        self.dim = op_num
        self.static_entities = []
        self.able_merge_list = merge_op_pos
        if force_merge:
            entity = [0] * op_num
            for pos in merge_op_pos:
                entity[pos] = 1
            self.static_entities.append(entity)
        else:
            num_merge_ops = len(merge_op_pos)
            enums = gen_enum([1, 0], num_merge_ops)
            for enum in enums:
                entity = [0] * op_num
                for i in range(num_merge_ops):
                    entity[merge_op_pos[i]] = enum[i]
                self.static_entities.append(entity)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "merge"

    def __repr__(self):
        return f"MergeSpce(static_entities={self.static_entities}, able_merge_list={self.able_merge_list}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    def able_merge(self, pos):
        return pos in self.able_merge_list


class EnumSpace(SubSpace):
    def __init__(self, knobs):
        self.dim = 2
        self.static_entities = knobs
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]

    def __repr__(self):
        return f"EnumSpace(static_entities={self.static_entities}, dim={self.dim}, size={self.size}, num_direction={self.num_direction}, directions={self.directions})"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


def generate_inline_space(op_lst, down_graph, force_inline=False):
    inline_op_pos = []
    for i, op in enumerate(op_lst):
        if able_inline(op, down_graph):
            inline_op_pos.append(i)
    return InlineSpace(inline_op_pos, len(op_lst), force_inline=force_inline)


def generate_merge_space(op_lst, down_graph, force_merge=False):
    merge_ops = list(range(len(op_lst)))
    return MergeSpce(merge_ops, len(op_lst), force_merge=force_merge)


def generate_fuse_space(loops, groups):
    return FuseSpace(groups, loops)


def generate_split_space(extent, nparts, allow_non_divisible='off'):
    return SplitSpace(nparts, extent, allow_non_divisible=allow_non_divisible)


def generate_reorder_space(num_spatial_axis):
    return ReorderSpace(num_spatial_axis)


def generate_unroll_space(explicit=False):
    return UnrollSpace([0, 1, 512, 1500], explicit=explicit)


def generate_space_intra_op(op, down_graph, slevel=4, rlevel=3, groups=3, split_policy="off", unroll_policy="off", fuse_policy="fuse_spatial", reorder_policy="last"):
    spatial_axis_names = [x.var.name for x in op.axis]
    spatial_axis_extents = [x.dom.extent.value for x in op.axis]
    reduced_axis_names = [x.var.name for x in op.reduce_axis]
    reduced_axis_extents = [x.dom.extent.value for x in op.reduce_axis]

    ##############################################################
    # generate space: 
    schedule_space = Space()

    # - fuse space
    if fuse_policy == "fuse_spatial":
        fuse_space = generate_fuse_space(spatial_axis_names, groups)
        schedule_space.add_subspace("fuse_spatial", fuse_space, "fuse")

    # - split space
    for i, (name, extent) in enumerate(zip(spatial_axis_names, spatial_axis_extents)):
        split_space = generate_split_space(extent, slevel, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "spatial")
    for i, (name, extent) in enumerate(zip(reduced_axis_names, reduced_axis_extents)):
        split_space = generate_split_space(extent, rlevel, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "reduce")

    # - reorder space
    if reorder_policy == "last":
        reorder_space = generate_reorder_space(groups)
        schedule_space.add_subspace("reorder", reorder_space, "reorder")

    # -unroll space
    unroll_space = generate_unroll_space(explicit=(unroll_policy == "explicit"))
    schedule_space.add_subspace("unroll", unroll_space, "unroll")
    
    # - other special spaces can be added   

    return schedule_space


def generate_space_inter_op(op_lst, down_graph, force_inline=False, force_merge=False, special_space=None):

    ##############################################################
    # generate space:
    schedule_space = Space()
    # - inline space
    inline_space = generate_inline_space(op_lst, down_graph, force_inline=force_inline)
    schedule_space.add_subspace("inline", inline_space, "inline")
    # - merge space
    # merge_space = generate_merge_space(op_lst, down_graph, force_merge=force_merge)
    # schedule_space.add_subspace("merge", merge_space, "merge")
    
    # - other special spaces can be added   
    special_space = {} if special_space is None else special_space
    for key, sspace in special_space.items():
        schedule_space.add_subspace(key, sspace, "special")

    return schedule_space