import tvm
import re

def create_target(tgt_dev, tgt_host, tgt_options=None):
    def process_target(tgt_str):
        if tgt_str not in tvm.target.Target.list_kinds():
            # try target tag:
                if tgt_str in tvm.target.tag.list_tags().keys():
                    tgt_str = str(tvm.target.Target(tgt_str))
                else:
                    raise ValueError("Target not a tvm target")
        cdg = []
        for t in re.findall((
            r"(\-{0,2}[\w\-]+\=?"
            r"(?:[\w\+\-\.]+(?:,[\w\+\-\.])*"
            r"|[\'][\w\+\-,\s\.]+[\']"
            r"|[\"][\w\+\-,\s\.]+[\"])*"
            r"|,)"
        ), tgt_str):
            cdg.append(t)
        t = " ".join(cdg)
        oa = cdg[1:] if len(cdg) > 1 else []
        ops = {}
        for op in oa:
            if op.startswith("--"):
                ops[op[2:]] = True
            else:
                opt = op[1:] if op.startswith("-") else opt
                on, ov = op.split("=", maxsplit=1)
                ov = ov[1:-1] if ov[0] in ('"', "'") else ov
                ops[on] = ov
        return (cdg[0], ops, t)

    def update(tgt_proc, options):
        if options is None:
            return tgt_proc
        if tgt_proc[0] in options:
            tgt_proc[1].update(options[tgt_proc[0]])
        return tgt_proc

    def tgt_to_def(tgt_proc):
        return "{0} {1}".format(tgt_proc[0], " ".join([f"-{k}={v}" for k, v in tgt_proc[1].items()]))

    tgt_dev_proc = tgt_to_def(update(process_target(tgt_dev), tgt_options))
    tgt_host_proc = tgt_to_def(update(process_target(tgt_host), tgt_options)) if tgt_host is not None else None

    tgt_ret = tvm.target.Target(tgt_dev_proc, host=tgt_host_proc)

    return tvm.target.Target.check_and_update_host_consist(tgt_ret, tgt_host)









