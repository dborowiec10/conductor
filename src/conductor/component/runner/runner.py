from conductor.component.runner.local import LocalRunner
from conductor.component.runner.remote_tvm import RemoteTVMRunner
from conductor.component.runner.remote_doppler import RemoteDopplerRunner

                
runners = {
    "local": LocalRunner,
    "remote_tvm": RemoteTVMRunner,
    "remote_doppler": RemoteDopplerRunner
}