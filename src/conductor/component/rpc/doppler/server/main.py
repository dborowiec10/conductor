from __future__ import absolute_import, print_function
from absl import app
from absl import flags
import logging
import threading
import tvm
import rpyc
from rpyc.utils.server import ThreadedServer
import socket
from conductor.component.runner.remote_doppler_server import RemoteDopplerRunnerServer

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
rpyc.core.protocol.DEFAULT_CONFIG['allow_public_attrs'] = True
rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout'] = None


logger = logging.getLogger("conductor.component.rpc.doppler.server")

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "0.0.0.0", "host ip address for server")
flags.DEFINE_integer("port", 9000, "host port for server")
flags.DEFINE_integer("port_end", 9199, "end host port for server")
flags.DEFINE_string("tracker_host", "0.0.0.0", "host ip address for the tracker")
flags.DEFINE_string("name", "unimachine", "name/id of the machine this server will run on")
flags.DEFINE_string("device", "v100", "type of device")
flags.DEFINE_list("dev_ids", [0], "idxs of the gpu to use, e.g. 0,1,2,3")

class ServerService(rpyc.Service):
    def __init__(self, host, port, trckr_host, trckr_port, name, device, dev_ids):
        self.host = host
        self.port = port
        self.trckr_host = trckr_host
        self.trckr_port = trckr_port
        self.trckr_conn = rpyc.connect(trckr_host, trckr_port)
        self.name = name
        self.device_name = device
        self.device_ids = dev_ids
        self.sessions = {}
        self.sessions_by_dev = {}
        
    def register_with_tracker(self):
        self.key = self.trckr_conn.root.register_server(
            self.host,
            self.port,
            self.name,
            [self.device_name + "." + str(x) for x in self.device_ids]
        )
        logger.info("Registered with tracker, key: " + self.key)
      
    def on_connect(self, conn):
        pass
        
    def on_disconnect(self, conn):
        pass
        
    def exposed_get_sm(self, dev_id):
        t = self.sessions_by_dev[self.device_name + "." + str(dev_id)].target
        def sm_from_ctx(ctx, tgt):
            if ctx.exist:
                if ("cuda" in tgt.keys or "opencl" in tgt.keys or "rocm" in tgt.keys or "vulkan" in tgt.keys):
                    sm = ("sm_" + "".join(ctx.compute_version.split(".")))
                else:
                    sm = None
            else:
                raise RuntimeError("Could not find a context!")
            return sm
        if isinstance(t, str):
            tgt = tvm.target.Target(t)
        else:
            tgt = t
        if "cuda" in tgt.keys:
            ctx = tvm.cuda(dev_id)
        else:
            ctx = tvm.cpu(dev_id)
        return sm_from_ctx(ctx, tgt)
    
    def exposed_get_build_options(self, dev_id):
        t = self.sessions_by_dev[self.device_name + "." + str(dev_id)].target
        def options_from_ctx(ctx, tgt):
            options = {}
            if ctx.exist:
                if ("cuda" in tgt.keys or "opencl" in tgt.keys or "rocm" in tgt.keys or "vulkan" in tgt.keys):
                    max_dims = ctx.max_thread_dimensions
                    options["check_gpu"] = {
                        "max_shared_memory_per_block": ctx.max_shared_memory_per_block,
                        "max_threads_per_block": ctx.max_threads_per_block,
                        "max_thread_x": max_dims[0],
                        "max_thread_y": max_dims[1],
                        "max_thread_z": max_dims[2],
                    }
            else:
                raise RuntimeError("Could not find a context!")
            return options
        if isinstance(t, str):
            tgt = tvm.target.Target(t)
        else:
            tgt = t
        if "cuda" in tgt.keys:
            ctx = tvm.cuda(dev_id)
        else:
            ctx = tvm.cpu(dev_id)
        options = options_from_ctx(ctx, tgt)
        return options
        
    def exposed_add_session(self, key, sess_data):
        logger.info("Adding session: " + key)
        self.sessions[key] = RemoteDopplerRunnerServer(sess_data["devices"])
        for d in sess_data["devices"]:
            self.sessions_by_dev[d] = self.sessions[key]
        logger.info("Added session: " + key)
        return True
    
    def exposed_remove_session(self, key):
        logger.info("Removing session: " + key)
        for d in self.sessions[key].devices:
            del self.sessions_by_dev[d]
        self.sessions[key].unload()
        del self.sessions[key]
        logger.info("Removed session: " + key)
        return True

    def exposed_ping(self):
        return "pong"
    
    def exposed_client_ping(self, key):
        if key in self.sessions:
            return "pong"
        else:
            logger.warning("Invalid session key attempted to client ping")
            return None

    def _check_and_run(self, sesskey, method, *args):
        if sesskey not in self.sessions:
            logger.error("Invalid session key")
            return None, False
        else:
            return (method(*args), True)
    
    def exposed_r_open_file(self, sesskey, filename, mode):
        return self._check_and_run(sesskey, self.sessions[sesskey].open_file, filename, mode)
    
    def exposed_r_set_evaluator(self, sesskey, evaluator_type, evaluator_config):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_evaluator, evaluator_type, evaluator_config)
    
    def exposed_r_set_run_type(self, sesskey, run_type):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_run_type, run_type)
    
    def exposed_r_set_n_parallel(self, sesskey, n_parallel):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_n_parallel, n_parallel)
    
    def exposed_r_set_timeout(self, sesskey, timeout):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_timeout, timeout)
    
    def exposed_r_set_remove_built_schedule(self, sesskey, remove_built_schedule):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_remove_built_schedule, remove_built_schedule)
    
    def exposed_r_set_cooldown_interval(self, sesskey, cooldown_interval):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_cooldown_interval, cooldown_interval)
    
    def exposed_r_set_target(self, sesskey, target):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_target, target)
    
    def exposed_r_set_target_host(self, sesskey, target_host):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_target_host, target_host)
    
    def exposed_r_set_orch_scheduler(self, sesskey, orch_scheduler_type):
        return self._check_and_run(sesskey, self.sessions[sesskey].set_orch_scheduler, orch_scheduler_type)
    
    def exposed_r_load(self, sesskey):
        return self._check_and_run(sesskey, self.sessions[sesskey].load)
        
    def exposed_r_unload(self, sesskey):
        return self._check_and_run(sesskey, self.sessions[sesskey].unload)
    
    def exposed_r_load_workers(self, sesskey, num_devs):
        return self._check_and_run(sesskey, self.sessions[sesskey].load_workers, num_devs)
    
    def exposed_r_load_one_per_dev(self, sesskey):
        return self._check_and_run(sesskey, self.sessions[sesskey].load_one_per_dev),
    
    def exposed_r_run(self, sesskey, configs, build_results, device_ids, files):
        return self._check_and_run(sesskey, self.sessions[sesskey].run, configs, build_results, device_ids, files)

    def exposed_r_profiling_checkpoint(self, sesskey, checkpoint, ctx):
        return self._check_and_run(sesskey, self.sessions[sesskey].rprofiling_checkpoint, checkpoint, ctx)

    def exposed_r_init_profilers(self, sesskey, profilers_specs):
        return self._check_and_run(sesskey, self.sessions[sesskey].exp_init_profilers, profilers_specs)

    def exposed_r_init_experiment(self, sesskey, store_config, exp_id):
        return self._check_and_run(sesskey, self.sessions[sesskey].exp_init_experiment, store_config, exp_id)

    def exposed_r_sync_experiment(self, sesskey):
        return self._check_and_run(sesskey, self.sessions[sesskey].exp_sync_experiment)
    
    def exposed_r_exp_set_task(self, sesskey, task_idx):
        return self._check_and_run(sesskey, self.sessions[sesskey].exp_set_task, task_idx)
    
    def exposed_r_exp_set_method(self, sesskey, method_idx):
        return self._check_and_run(sesskey, self.sessions[sesskey].exp_set_method, method_idx)
    
    def exposed_r_exp_set_experiment_stage(self, sesskey, stage):
        return self._check_and_run(sesskey, self.sessions[sesskey].exp_set_experiment_stage, stage)

def establish_port(host, port, port_end):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    curp = port
    while curp <= port_end:
        try:
            sock.bind((host, curp))
            sock.close()
            return curp
        except Exception as e:
            curp += 1
    sock.close()
    logger.error("Unable to find suitable port to bind to")
    return -1

def find_tracker(thost, ps, pe):
    curp = ps
    conn = None
    while curp <= pe:
        try:
            conn = rpyc.connect(thost, curp)
            r = conn.root.ping()
            if r == "pong":
                conn.close()
                return curp
            else:
                curp += 1

        except:
            curp += 1
    conn.close()
    logger.error("Unable to find tracker")
    return -1

def main(argv):
    del argv
    prt = establish_port(FLAGS.host, FLAGS.port, FLAGS.port_end)
    trckprt = find_tracker(FLAGS.tracker_host, FLAGS.port, FLAGS.port_end)
    if trckprt == -1:
        logger.error("Unable to find tracker")
        return
    logger.info("Found tracker at: " + FLAGS.tracker_host + ":" + str(trckprt))
    srv = ServerService(FLAGS.host, prt, FLAGS.tracker_host, trckprt, FLAGS.name, FLAGS.device, FLAGS.dev_ids)
    t = ThreadedServer(srv, hostname=FLAGS.host, port=prt)
    logger.info("Bound server to: " + FLAGS.host + ":" + str(prt))
    thr = threading.Thread(target=t.start)
    thr.start()
    srv.register_with_tracker()
    thr.join()
    
if __name__ == '__main__':
    app.run(main)
    
