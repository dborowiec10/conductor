from tvm.ir.transform import Sequential, PassContext
from tvm.relay.transform import RemoveUnusedFunctions, ConvertLayout
from tvm.relay import build

from conductor.compiler._target import create_target
from conductor.mediation import FallbackContext, HistoryBestContext, TophubContext
from conductor.model import CompiledModel
from conductor.component.scheduler.scheduler import schedulers

import traceback

def compile_routine(_input):
    model, options = _input
    try:
        model._load()
        target, _ = create_target(options.target, options.target_host, options.target_opts)
        mod, params = (model.get_json(), model.get_params())
        if options.layouts is not None:
            seq = Sequential(
                [
                    RemoveUnusedFunctions(),
                    ConvertLayout(options.layouts),
                ]
            )
            with PassContext(opt_level=options.opt_level):
                mod = seq(mod)

        disabled_pass = None if len(options.disabled_passes) < 1 else options.disabled_passes
        required_pass = None if len(options.required_passes) < 1 else options.required_passes
        disabled_pass = {"AutoSchedulerLayoutRewrite"}

        if options.implementations_type == None:
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = False
            with FallbackContext():
                with PassContext(
                    opt_level=options.opt_level, 
                    config=_config,
                    disabled_pass=disabled_pass,
                    required_pass=required_pass
                ):
                    module = build(mod, target=target, params=params)

        elif options.implementations_type == "tophub":
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = False
            with TophubContext(target):
                with PassContext(
                    opt_level=options.opt_level, 
                    config=_config,
                    disabled_pass=disabled_pass,
                    required_pass=required_pass
                ):
                    module = build(mod, target=target, params=params)

        elif options.implementations_type == "sketch":
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = True
            ctx = HistoryBestContext(options.implementations_path)
            ctx.register_scheduler(schedulers[options.implementations_type]())
            with ctx:
                with PassContext(
                    opt_level=options.opt_level, 
                    config=_config,
                    disabled_pass=disabled_pass,
                    required_pass=required_pass
                ):
                    module = build(mod, target=target, params=params)

        elif options.implementations_type == "template":
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = False
            with HistoryBestContext(options.implementations_path):
                with PassContext(
                    opt_level=options.opt_level, 
                    config=_config,
                    disabled_pass=disabled_pass,
                    required_pass=required_pass
                ):
                    module = build(mod, target=target, params=params)

        elif options.implementations_type == "flex":
            _config = options.pass_configurations
            _config["relay.backend.use_auto_scheduler"] = True
            ctx = HistoryBestContext(options.implementations_path)
            ctx.register_scheduler(schedulers[options.implementations_type]())
            # TODO: needs doing
            pass
        else:
            raise RuntimeError("Invalid implementation type")

        cm = CompiledModel(
            model.model_name,
            model.model_framework,
            model.model_batch_size,
            options.opt_level,
            options.instr_count,
            options.implementations_type,
            options.implementations_path,
            model.model_in_shape,
            model.model_out_shape,
            module.get_lib(),
            module.get_graph_json(),
            module.get_params(),
            custom_path=options.custom_path,
            save_source=options.save_source,
            save_model=options.save_model
        )
        
        cm.save()
        return (cm, None)

    except Exception as e:
        return (None, traceback.format_exc())
    


