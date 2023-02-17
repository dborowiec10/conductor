import time
from conductor.worker.worker import StatusKind
from conductor.mediation import MeasureErrorNo

def prep_work(configs, build_results, device_ids):
    unique_dev_ids = list(set(device_ids))
    work_by_dev = {di: [] for di in unique_dev_ids}
    for k, (c, br, di) in enumerate(zip(configs, build_results, device_ids)):
        work_by_dev[di].append((k, c, br, di))
    left_to_do = len(build_results)
    return left_to_do, unique_dev_ids, work_by_dev

def select_sub_batch(left_to_do, unique_dev_ids, work_by_dev, n_parallel=None):
    pw = []
    ltd = left_to_do
    for di in unique_dev_ids:
        nleft = len(work_by_dev[di])
        # None signifies we execute in parallel all of the configs
        if n_parallel == None:
            pw += work_by_dev[di]
            work_by_dev[di] = []
            ltd -= nleft
        else:
            # otherwise we calculate number of required configs for this batch 
            num_items = n_parallel if n_parallel <= nleft else nleft
            if num_items > 0:
                pw += work_by_dev[di][:num_items]
                work_by_dev[di] = work_by_dev[di][num_items:]
            ltd -= num_items
    return pw, ltd, unique_dev_ids

def handle_build_error(bld, callback):
    costs = [1e20]
    mean = 1e20
    total_time = 1e20
    status = "B"
    achieved_flop = 0
    r = (costs, bld.error_no, bld.error_msg, bld.time_cost, time.time(), achieved_flop, mean, total_time, status)
    callback((None, "build_error", (None, bld.error_no, None, None, None, mean, total_time, None, achieved_flop)))
    return r, status

def handle_result(worker, btc, timeout):
    status, err_msg, res = worker.get()
    if status != StatusKind.COMPLETE:
        time_cost = timeout + btc
        timestamp = time.time()
        error_msg = err_msg
        costs = [1e20]
        mean = 1e20
        total_time = 1e20
        achieved_flop = 0
        if status == StatusKind.TIMEOUT:
            error_no = MeasureErrorNo.RUN_TIMEOUT
            measure_status = "T"
        else:
            error_no = MeasureErrorNo.RUNTIME_DEVICE
            measure_status = "E"
    else:
        costs, error_no, error_msg, time_cost, timestamp, mean, total_time, _, achieved_flop = res
        measure_status = "*" if error_no == MeasureErrorNo.NO_ERROR else "E"
    _result = (costs, error_no, error_msg, time_cost, timestamp, achieved_flop, mean, total_time, measure_status)
    _status = measure_status
    return (_result, _status)