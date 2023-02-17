import multiprocessing as _multi

from conductor.profiler.powercap_profiler import PowercapProfiler
import time

stime = time.time()

n_parallel = 5

def handler(a, b, c, d):
    # print(a, b, c, d)
    time.sleep(5)   
    return (a + b + c, d)

other_header = ["oo", "aa", "hh"]

timeprof = PowercapProfiler("builder", ".")
timeprof.begin(other_header)

def callback(res):
    pass
    # print("CALLBACK", res, flush=True)
    

k = timeprof.start(["o+", "a-"])
timeprof.append_context(k, ["h+"])

result_futures = []
with _multi.Pool(n_parallel) as pool:
    for i in range(5):
        
        
        
        h = pool.apply_async(handler, [1, 2, 3, k], callback=callback)
        result_futures.append(h)
    

    for r in result_futures:
        r.get()
        # print("RES FUT", r.get())

timeprof.stop(k)
timeprof.persist()
timeprof.end()

print("time taken", time.time() - stime)
