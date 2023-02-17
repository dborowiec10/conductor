from tvm.ir.transform import PassContext
from tvm.driver.build_module import build

from conductor.compiler._target import create_target
from conductor.mediation import FallbackContext, HistoryBestContext, TophubContext
from conductor.tensor_program import CompiledTensorProgram
from conductor.component.scheduler.flex import FlexScheduler
from conductor.component.scheduler.sketch import SketchScheduler
from conductor.component.scheduler.template import TemplateScheduler

import traceback

def compile_routine(_input):
    tp, options = _input
    
    try:
        target, _ = create_target(options.target, options.target_host, options.target_opts)
        tp._load()

        def build_tp(schedule, args, error_no, error_msg, target, _config: dict):
            with target:
                with PassContext(config=_config):
                    return build(schedule, args, target=target)

        if options.implementations_type == None:
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = False
            if tp.is_templateable():
                scheduler = TemplateScheduler()
                sched, args, error_no, error_msg = scheduler.from_tensor_program_baseline(tp)
                outfunc = build_tp(sched, args, error_no, error_msg, target, _config)
            else:
                scheduler = SketchScheduler()
                sched, args, error_no, error_msg = scheduler.from_tensor_program_baseline(tp, options=options)
                outfunc = build_tp(sched, args, error_no, error_msg, target, _config)

        elif options.implementations_type == "tophub":
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = False
            if not tp.is_templateable():
                raise RuntimeError("Cannot apply tophub configuration to non-templateable schedulable!")
            else:
                scheduler = TemplateScheduler()
                sched, args, error_no, error_msg = scheduler.from_tensor_program_context(tp, TophubContext(target))
                outfunc = build_tp(sched, args, error_no, error_msg, target, _config)

        elif options.implementations_type == "sketch":
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = True
            scheduler = SketchScheduler()
            sched, args, error_no, error_msg = scheduler.from_tensor_program_context(tp, HistoryBestContext(options.implementations_path))
            outfunc = build_tp(sched, args, error_no, error_msg, target, _config)

        elif options.implementations_type == "template":
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = False
            scheduler = TemplateScheduler()
            sched, args, error_no, error_msg = scheduler.from_tensor_program_context(tp, HistoryBestContext(options.implementations_path))
            outfunc = build_tp(sched, args, error_no, error_msg, target, _config)

        elif options.implementations_type == "flex":
            # TODO: needs doing
            pass
        else:
            raise RuntimeError("Invalid implementation type")

        ctm = CompiledTensorProgram(
            tp.identifier,
            args,
            tp.func_name,
            options.instr_count,
            tp.is_topi,
            options.implementations_type,
            options.implementations_path,
            outfunc,
            custom_path=options.custom_path,
            save_source=options.save_source,
            save_tensor_program=options.save_tensor_program
        )

        ctm.save()
        return (ctm, None)

    except Exception as e:
        return (None, traceback.format_exc())