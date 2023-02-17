import multiprocessing as _multi

from conductor.profiler.time_profiler import TimeProfiler

n_parallel = 5

def handler(a, b, c, d):
    print(a, b, c, d)
    return (a + b + c, d)

other_header = ["oo", "aa", "hh"]

timeprof = TimeProfiler("builder", ".")
timeprof.begin(other_header)

def callback(res):
    print("CALLBACK", res, flush=True)
    timeprof.stop(res[1])


result_futures = []
with _multi.Pool(n_parallel) as pool:
    for i in range(5):
        k = timeprof.start(["o+", "a-"])
        timeprof.append_context(k, ["h+"])
        
        h = pool.apply_async(handler, [1, 2, 3, k], callback=callback)
        result_futures.append(h)
    

    for r in result_futures:
        print("RES FUT", r.get())

timeprof.persist()
timeprof.end()
