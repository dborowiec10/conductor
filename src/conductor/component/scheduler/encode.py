import tvm

@tvm._ffi.register_object
class IterVarAttr(tvm.runtime.Object):
    """IterVarAttr axis."""

def handle_SourceName(sb, storage):
    s_name = look_call(sb.name, storage)
    return s_name

def handle_Span(sb, storage):
    s_source_name = look_call(sb.source_name, storage)
    s_line = look_call(sb.line, storage)
    s_column = look_call(sb.column, storage)
    s_end_line = look_call(sb.end_line, storage)
    s_end_column = look_call(sb.end_column, storage)
    return "%s%s%s%s%s" % (s_source_name, s_line, s_column, s_end_line, s_end_column)

def handle_Any(sb, storage):
    s_span = look_call(sb.span, storage)
    return "%s" % (s_span)

def handle_Shuffle(sb, storage):
    s_vectors = look_call(sb.vectors, storage)
    s_indices = look_call(sb.indices, storage),
    s_span = look_call(sb.span, storage)
    return "%s%s%s" % (s_vectors, s_indices, s_span)

def handle_Call(sb, storage):
    s_op = look_call(sb.op, storage)
    s_args = look_call(sb.args, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s" % (s_span, s_op, s_args)

def handle_Let(sb, storage):
    s_var = look_call(sb.var, storage)
    s_value = look_call(sb.value, storage)
    s_body = look_call(sb.body, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s%s" % (s_var, s_value, s_body, s_span)

def handle_Broadcast(sb, storage):
    s_value = look_call(sb.value, storage)
    s_lanes = look_call(sb.lanes, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s" % (s_value, s_lanes, s_span)

def handle_Ramp(sb, storage):
    s_base = look_call(sb.base, storage)
    s_stride = look_call(sb.stride, storage)
    s_lanes = look_call(sb.lanes, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s%s" % (s_base, s_stride, s_lanes, s_span)

def handle_BufferLoad(sb, storage):
    s_buffer = look_call(sb.buffer, storage)
    s_indices = look_call(sb.indices, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s" % (s_buffer, s_indices, s_span)

def handle_Load(sb, storage):
    s_buffer_var = look_call(sb.buffer_var, storage)
    s_index = look_call(sb.index, storage)
    s_predicate = look_call(sb.predicate, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s%s" % (s_buffer_var, s_index, s_predicate, s_span)

def handle_Select(sb, storage):
    s_condition = look_call(sb.condition, storage)
    s_true_value = look_call(sb.true_value, storage)
    s_false_value = look_call(sb.false_value, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s%s" % (s_condition, s_true_value, s_false_value, s_span)

def handle_Not(sb, storage):
    s_a = look_call(sb.a, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s" % (s_a, s_span)

def handle_simple_dbl_param(sb, storage):
    s_type = look_call(sb.__class__, storage)
    s_a = look_call(sb.a, storage)
    s_b = look_call(sb.b, storage)
    s_span = look_call(sb.span, storage)
    fn_name = common_funcs[s_type]
    return "%s%s%s%s" % (fn_name, s_a, s_b, s_span)

def handle_Cast(sb, storage):
    s_value = look_call(sb.value, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s" % (s_value, s_span)

def handle_ProducerLoad(sb, storage):
    s_producer = look_call(sb.producer, storage)
    s_indices = look_call(sb.indices, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s" % (s_producer, s_indices, s_span)

def handle_CommReducer(sb, storage):
    s_identity_element = look_call(sb.identity_element, storage)
    s_result = look_call(sb.result, storage)
    s_lhs = look_call(sb.lhs, storage)
    s_rhs = look_call(sb.rhs, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s%s%s" % (s_identity_element, s_result, s_lhs, s_rhs, s_span)

def handle_Reduce(sb, storage):
    s_value_index = look_call(sb.value_index, storage)
    s_condition = look_call(sb.condition, storage)
    s_combiner = look_call(sb.combiner, storage)
    s_source = look_call(sb.source, storage)
    s_init = look_call(sb.init, storage)
    s_axis = look_call(sb.axis, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s%s%s%s%s" % (s_value_index, s_condition, s_combiner, s_source, s_init, s_axis, s_span)

def handle_Range(sb, storage):
    s_min = look_call(sb.min, storage)
    s_extent = look_call(sb.extent, storage)
    return "%s%s" % (s_min, s_extent)

def handle_Var(sb, storage):
    s_name = look_call(sb.name, storage)
    s_type_annotation = look_call(sb.type_annotation, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s" % (s_name, s_type_annotation, s_span)

def handle_SizeVar(sb, storage):
    s_name = look_call(sb.name, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s" % (s_name, s_span)

def handle_IterVar(sb, storage):
    s_dom = look_call(sb.dom, storage)
    s_iter_type = look_call(sb.iter_type, storage)
    s_thread_tag = look_call(sb.thread_tag, storage)
    s_var = look_call(sb.var, storage)
    s_span = look_call(sb.span, storage)
    return "%s%s%s%s%s" % (s_dom, s_iter_type, s_thread_tag, s_var, s_span)

def handle_ComputeOp(sb, storage):
    if str(sb) in storage["ops"]:
        return storage["ops"][str(sb)]
    else:
        storage["ops"][str(sb)] = "parent"
    s_axis = look_call(sb.axis, storage)
    s_body = look_call(sb.body, storage)
    s_input_tensors = look_call(sb.input_tensors, storage)
    s_name = look_call(sb.name, storage)
    s_num_outputs = look_call(sb.num_outputs, storage)
    s_tag = look_call(sb.tag, storage)
    s_reduce_axis = look_call(sb.reduce_axis, storage)
    s_output = look_call([sb.output(i) for i in range(int(sb.num_outputs))], storage)
    out = "%s%s%s%s%s%s%s%s" % (s_axis, s_body, s_input_tensors, s_name, s_num_outputs, s_tag, s_reduce_axis, s_output)
    storage["ops"][str(sb)] = out
    return out

def handle_FloatImm(sb, storage):
    return look_call(sb.value, storage)

def handle_IntImmEnum(sb, storage):
    return look_call(sb.value, storage)

def handle_IntImm(sb, storage):
    return look_call(sb.value, storage)

def handle_StringImm(sb, storage):
    return look_call(sb.value, storage)

def handle_Tensor(sb, storage):
    s_name = look_call(sb.name, storage)
    s_shape = look_call(sb.shape, storage)
    if str(sb) in storage["ops"]:
        s_op = storage["ops"][str(sb)]
    else:
        s_op = look_call(sb.op, storage)
        storage["ops"][str(sb)] = s_op
    return "%s%s%s" % (s_name, s_shape, s_op)

def handle_PlaceholderOp(sb, storage):
    if str(sb) in storage["ops"]:
        return storage["ops"][str(sb)]
    else:
        storage["ops"][str(sb)] = "p"
    s_name = look_call(sb.name, storage)
    s_tag = look_call(sb.tag, storage)
    s_shape = look_call(sb.shape, storage)
    s_num_outputs = look_call(sb.num_outputs, storage)
    s_input_tensors = look_call(sb.input_tensors, storage)
    s_output = look_call([sb.output(i) for i in range(int(sb.num_outputs))], storage)
    out = "%s%s%s%s%s%s" % (s_name, s_tag, s_shape, s_num_outputs, s_input_tensors, s_output)
    storage["ops"][str(sb)] = out
    return out

def handle_Array(a, storage):
    nm = ""
    if len(a) > 0:
        for aa in a:
            s_ax = look_call(aa, storage)
            nm += s_ax
    return nm

def handle_Map(m, storage):
    nm = ""
    if len(m.items()) > 0:
        for k, v in m.items():
            s_key = look_call(k, storage)
            s_val = look_call(v, storage)
            nm += s_key + s_val
    return nm

def handle_Split(sb, storage):
    s_parent = look_call(sb.parent, storage)
    s_outer = look_call(sb.outer, storage)
    s_inner = look_call(sb.inner, storage)
    s_factor = look_call(sb.factor, storage)
    s_nparts = look_call(sb.nparts, storage)
    return "%s%s%s%s%s" % (s_parent, s_outer, s_inner, s_factor, s_nparts)

def handle_Fuse(sb, storage):
    s_outer = look_call(sb.outer, storage)
    s_inner = look_call(sb.inner, storage)
    s_fused = look_call(sb.fused, storage)
    return "%s%s%s" % (s_outer, s_inner, s_fused)

def handle_IterVarAttr(sb, storage):
    s_iter_type = look_call(sb.iter_type, storage)
    s_bind_thread = look_call(sb.bind_thread, storage)
    s_prefetch_data = look_call(sb.prefetch_data, storage)
    s_prefetch_offset = look_call(sb.prefetch_offset, storage)
    s_tensor_intrin = look_call(sb.tensor_intrin, storage)
    s_dim_align_factor = look_call(sb.dim_align_factor, storage)
    s_dim_align_offset = look_call(sb.dim_align_offset, storage)
    s_pragma_keys = look_call(sb.pragma_keys, storage)
    s_pragma_values = look_call(sb.pragma_values, storage)
    return "%s%s%s%s%s%s%s%s%s" % (
        iter_types[int(s_iter_type)], s_bind_thread, s_prefetch_data, s_prefetch_offset,
        s_tensor_intrin, s_dim_align_factor, s_dim_align_offset,
        s_pragma_keys, s_pragma_values
    )

def handle_Stage(sb, storage):
    if str(sb.op) in storage["ops"]:
        s_op = storage["ops"][str(sb.op)]
    else:
        s_op = look_call(sb.op, storage)
        storage["ops"][str(sb.op)] = s_op
    if str(sb.origin_op) in storage["ops"]:
        s_origin_op = storage["ops"][str(sb.origin_op)]
    else:
        s_origin_op = look_call(sb.origin_op, storage)
        storage["ops"][str(sb.origin_op)] = s_origin_op
    s_all_iter_vars = look_call(sb.all_iter_vars, storage)
    s_leaf_iter_vars = look_call(sb.leaf_iter_vars, storage)
    s_relations = look_call(sb.relations, storage)
    s_iter_var_attrs = look_call(sb.iter_var_attrs, storage)
    s_attach_type = look_call(sb.attach_type, storage)
    s_attach_ivar = look_call(sb.attach_ivar, storage)
    s_attach_stage = look_call(sb.attach_stage, storage)
    s_scope = look_call(sb.scope, storage)
    s_is_output = look_call(sb.is_output, storage)
    s_group = look_call(sb.group, storage)
    s_num_child_stages = look_call(sb.num_child_stages, storage)
    return "%s%s%s%s%s%s%s%s%s%s%s%s%s" % (
        s_op, s_origin_op, s_all_iter_vars, s_leaf_iter_vars, s_relations, s_iter_var_attrs, s_attach_type,
        s_attach_ivar, s_attach_stage, s_scope, s_is_output, s_group, s_num_child_stages
    )

def look_call(val, storage):
    k__cls = str(val.__class__)
    if k__cls in lookup_table:
        return lookup_table[k__cls](val, storage)
    else:
        return "UNKNOWN LOOKUP KEY: " + k__cls

iter_types = ["DP", "TI", "CR", "OR", "DI", "UN", "VC", "PR", "TS"]

lookup_table = {
    str(type(None)): lambda x, y: "",
    str(str): lambda x, y: str(x),
    str(int): lambda x, y: str(x),
    str(float): lambda x, y: str(x),
    str(type): lambda x, y: str(x),
    str(list): handle_Array,
    str(tvm.runtime.container.String): lambda x, y: str(x),
    str(tvm.te.tensor.PlaceholderOp): handle_PlaceholderOp,
    str(tvm.te.tensor.ComputeOp): handle_ComputeOp,
    str(tvm.te.tensor.Tensor): handle_Tensor,
    str(tvm.te.schedule.Stage): handle_Stage,
    str(tvm.te.schedule.Split): handle_Split,
    str(tvm.te.schedule.Fuse): handle_Fuse,
    str(tvm.ir.container.Array): handle_Array,
    str(tvm.ir.container.Map): handle_Map,
    str(tvm.ir.expr.Range): handle_Range,
    str(tvm.ir.base.Span): handle_Span,
    str(tvm.ir.base.SourceName): handle_SourceName,
    str(tvm.tir.expr.IntImm): handle_IntImm,
    str(tvm.tir.expr.FloatImm): handle_FloatImm,
    str(tvm.tir.expr.IntImmEnum): handle_IntImmEnum,
    str(tvm.tir.expr.StringImm): handle_StringImm,
    str(tvm.tir.expr.IterVar): handle_IterVar,
    str(tvm.tir.expr.Var): handle_Var,
    str(tvm.tir.expr.SizeVar): handle_SizeVar,
    str(tvm.tir.expr.Reduce): handle_Reduce,
    str(tvm.tir.expr.CommReducer): handle_CommReducer,
    str(tvm.tir.expr.ProducerLoad): handle_ProducerLoad,
    str(tvm.tir.expr.Cast): handle_Cast,
    str(tvm.tir.expr.Mul): handle_simple_dbl_param,
    str(tvm.tir.expr.Add): handle_simple_dbl_param,
    str(tvm.tir.expr.Sub): handle_simple_dbl_param,
    str(tvm.tir.expr.Div): handle_simple_dbl_param,
    str(tvm.tir.expr.Mod): handle_simple_dbl_param,
    str(tvm.tir.expr.FloorMod): handle_simple_dbl_param,
    str(tvm.tir.expr.FloorDiv): handle_simple_dbl_param,
    str(tvm.tir.expr.Min): handle_simple_dbl_param,
    str(tvm.tir.expr.Max): handle_simple_dbl_param,
    str(tvm.tir.expr.EQ): handle_simple_dbl_param,
    str(tvm.tir.expr.NE): handle_simple_dbl_param,
    str(tvm.tir.expr.LT): handle_simple_dbl_param,
    str(tvm.tir.expr.LE): handle_simple_dbl_param,
    str(tvm.tir.expr.GT): handle_simple_dbl_param,
    str(tvm.tir.expr.GE): handle_simple_dbl_param,
    str(tvm.tir.expr.And): handle_simple_dbl_param,
    str(tvm.tir.expr.Or): handle_simple_dbl_param,
    str(tvm.tir.expr.Not): handle_Not,
    str(tvm.tir.expr.Select): handle_Select,
    str(tvm.tir.expr.Load): handle_Load,
    str(tvm.tir.expr.BufferLoad): handle_BufferLoad,
    str(tvm.tir.expr.Ramp): handle_Ramp,
    str(tvm.tir.expr.Broadcast): handle_Broadcast,
    str(tvm.tir.expr.Let): handle_Let,
    str(tvm.tir.expr.Call): handle_Call,
    str(tvm.tir.expr.Shuffle): handle_Shuffle,
    str(tvm.tir.expr.Any): handle_Any,
    str(IterVarAttr): handle_IterVarAttr
}

common_funcs = {
    str(tvm.tir.expr.Mul): "*",
    str(tvm.tir.expr.Add): "+",
    str(tvm.tir.expr.Sub): "-",
    str(tvm.tir.expr.Div): "/",
    str(tvm.tir.expr.Mod): "m",
    str(tvm.tir.expr.FloorMod): "fm",
    str(tvm.tir.expr.FloorDiv): "f/",
    str(tvm.tir.expr.Min): "mi",
    str(tvm.tir.expr.Max): "mx",
    str(tvm.tir.expr.EQ): "=",
    str(tvm.tir.expr.NE): "!=",
    str(tvm.tir.expr.LT): "<",
    str(tvm.tir.expr.LE): "<=",
    str(tvm.tir.expr.GT): ">",
    str(tvm.tir.expr.GE): ">=",
    str(tvm.tir.expr.And): "&",
    str(tvm.tir.expr.Or): "||"
}