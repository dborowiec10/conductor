import subprocess
import os
import re
import tempfile
import psutil
import signal

MPS_CTRL_PROG = 'nvidia-cuda-mps-control'
NVIDIA_SMI_PROG = 'nvidia-smi'

def findProcessIdByName(processName):
    listOfProcessObjects = []
    for proc in psutil.process_iter():
       try:
           pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
           if processName.lower() in pinfo['name'].lower():
               listOfProcessObjects.append(pinfo)
       except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
           pass
    return listOfProcessObjects

def comma_sep_device_ids(dev_list):
    s = ""
    for k, d in enumerate(dev_list):
        s += str(d)
        if k < (len(dev_list) - 1):
            s += ","
    return s

class NvidiaMPS(object):
    _name = "nvidia_mps"

    def __repr__(self):
        return NvidiaMPS._name

    def __init__(self, device_ids):
        self.device_ids = device_ids
        self.is_on = False

    def get_mps_procs(self):
        return findProcessIdByName(MPS_CTRL_PROG) + findProcessIdByName("nvidia-cuda-mps-server")

    def get_proc_env(self, pid):
        try:
            f = open('/proc/%i/environ' % pid, 'r')
        except:
            return ''
        else:
            return f.read().replace('\0', '\n')    
    
    def get_mps_dir(self, pid):
        data = self.get_proc_env(pid)
        r = re.search('^CUDA_MPS_PIPE_DIRECTORY=(.*)$', data, re.MULTILINE)
        try:
            return r.group(1)
        except:
            return None

    def start(self, mps_dir=None):
        if mps_dir is None:
            mps_dir = tempfile.mkdtemp()
        env = os.environ
        env['CUDA_MPS_PIPE_DIRECTORY'] = mps_dir
        env['CUDA_MPS_LOG_DIRECTORY'] = mps_dir
        env['CUDA_VISIBLE_DEVICES'] = comma_sep_device_ids(self.device_ids)
        p_smi = subprocess.Popen([NVIDIA_SMI_PROG, '-i', comma_sep_device_ids(self.device_ids), '-c', 'EXCLUSIVE_PROCESS'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        try:
            out = p_smi.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            self.is_on = False
            print("timeout running nvidia smi: START")
            pass
        p = subprocess.Popen([MPS_CTRL_PROG, '-d'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        try:
            out = p.communicate(timeout=2)[0]
            self.is_on = True
        except subprocess.TimeoutExpired:
            self.is_on = False
            out = "".encode("utf-8")
            print('timeout starting nvidia mps')
            pass
        if 'An instance of this daemon is already running' in out.decode("utf-8"):
            self.is_on = True
            print('running daemon already using %s' % mps_dir)
            
        return mps_dir

    def stop(self, mps_dir):
        if mps_dir == None:
            procs = self.get_mps_procs()
            for p in procs:
                try:
                    os.kill(p["pid"], signal.SIGKILL)
                    print("KILLED")
                except Exception as e:
                    print(e)
                    print("MPS Not running... but listed...", str(p))
            try:
                p = subprocess.Popen(['rmmod', 'nvidia_uvm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out = p.communicate(timeout=5)
                # print(out[0].decode("utf-8"))
                print("REMOVED NVIDIA MODULE")
            except Exception as e:
                print(e, flush=True)

            try:
                p2 = subprocess.Popen(['modprobe', 'nvidia_uvm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out = p2.communicate(timeout=5)
                # print(out[0].decode("utf-8"))
                print("REINSERTED NVIDIA MODULE")
            except Exception as e:
                print(e, flush=True)

        else:
            env = os.environ
            env['CUDA_MPS_PIPE_DIRECTORY'] = mps_dir
            env['CUDA_MPS_LOG_DIRECTORY'] = mps_dir
            p = subprocess.Popen([MPS_CTRL_PROG], stdin=subprocess.PIPE)
            try:
                p.communicate(b'quit\n', timeout=5)
            except subprocess.TimeoutExpired:
                print("timeout stopping nvidia mps")
            p_smi = subprocess.Popen([NVIDIA_SMI_PROG, '-i', comma_sep_device_ids(self.device_ids), '-c', 'DEFAULT'])
            try:
                p_smi.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                print('timeout running nvidia smi: END')

            procs = self.get_mps_procs()
            for p in procs:
                try:
                    os.kill(p["pid"], signal.SIGKILL)
                    print("KILLED")
                except:
                    print("MPS Not running... but listed...", str(p))

            try:
                p = subprocess.Popen(['rmmod', 'nvidia_uvm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out = p.communicate(timeout=5)
                print("REMOVED NVIDIA MODULE")
            except Exception as e:
                print(e, flush=True)

            try:
                p2 = subprocess.Popen(['modprobe', 'nvidia_uvm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out = p2.communicate(timeout=5)
                print("REINSERTED NVIDIA MODULE")
            except Exception as e:
                print(e, flush=True)
        self.is_on = False