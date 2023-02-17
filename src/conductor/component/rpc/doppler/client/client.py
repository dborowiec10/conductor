import rpyc
import logging
from rpyc.utils.server import ThreadedServer
import threading
logger = logging.getLogger("conductor.component.rpc.doppler.client")

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
rpyc.core.protocol.DEFAULT_CONFIG['allow_public_attrs'] = True
rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout'] = None

class ClientService(rpyc.Service):
    def exposed_ping(self):
        return "pong"

class ClientSession():
    def __init__(self, key, info):
        self.host = info["host"]
        self.port = info["port"]
        self.srv_name = info["srv_name"]
        self.devices = info["devices"]
        self.unique_id = info["unique_id"]
        self.key = key
        self.c = rpyc.connect(self.host, self.port)
        self.sms = {}
        self.build_options = {}
        try:
            ret = self.c.root.client_ping(self.key)
            if ret != "pong":
                raise Exception("Unable to connect to server")
        except Exception as e:
            logger.info(e)
            raise Exception("Unable to connect to server")

class Client():
    def __init__(self, client_host, client_port, tracker_host, tracker_port, devices):
        self.tracker_host = tracker_host
        self.tracker_port = tracker_port
        self.client_host = client_host
        self.client_port = client_port
        self.devices = {}
        self.tc = None
        self.sms = {}
        self.build_options = {}
        self.connect_tracker()
        unique_servers = self.get_unique_servers(devices)
        # set session per device (one session per server, one session may handle multiple devices)
        for ku, kv in unique_servers.items():
            key, info = self.tc.root.session(ku, kv, self.client_host, self.client_port)
            if key == None or info == None:
                raise Exception("Unable to get session")
            sess = ClientSession(key, info)
            for k in kv:
                self.devices[ku + "." + k] = sess
           
    def connect_tracker(self):
        # self.cs = ClientService()
        # self.cs_ts = ThreadedServer(self.cs, hostname=self.client_host, port=self.client_port)
        # self.cs_ts_t = threading.Thread(target=self.cs_ts.start, daemon=True).start()
        self.tc = rpyc.connect(self.tracker_host, self.tracker_port)
        try:
            ret = self.tc.root.ping()
            if ret != "pong":
                raise Exception("Unable to connect to tracker")
        except Exception as e:
            raise Exception("Unable to connect to tracker")
       
    def get_unique_servers(self, devices):
        unique_servers = {}
        for d in devices:
            spl = d.split(".")
            serv = spl[0]
            dev = spl[1]
            dev_idx = spl[2]
            if serv not in unique_servers:
                unique_servers[serv] = []
            unique_servers[serv].append(dev + "." + dev_idx)
        return unique_servers
    
    def get_unique_sessions(self, devices):
        unique_sessions = set()
        for d in devices:
            unique_sessions.add(self.devices[d])
        return unique_sessions 
                
    def session_by_device(self, device):
        return self.devices[device]
    
    def retrieve_sm(self, device, target):
        return self.devices[device].sms[device + "." + target]
    
    def get_sm(self, device, target):
        if device + "." + target in self.sms:
            return self.sms[device + "." + target]
        self.sms[device + "." + target] = self.devices[device].c.root.get_sm(int(device.split(".")[-1]))
        return self.sms[device + "." + target]
    
    def get_build_options(self, device, target):
        if device + "." + target in self.build_options:
            return self.build_options[device + "." + target]
        self.build_options[device + "." + target] = self.devices[device].c.root.get_build_options(int(device.split(".")[-1]))
        return self.build_options[device + "." + target]
    
    def call_all(self, method, *args, devices=None):
        unique_sessions = self.get_unique_sessions(self.devices)
        results = {}
        for s in unique_sessions:
            methd = rpyc.async_(getattr(s.c.root, method))
            a = (s.key,) + args
            results[s.key] = methd(*a)
        for s in unique_sessions:
            results[s.key].wait()
        out = []
        for s in unique_sessions:

            if results[s.key].ready:
                outval = results[s.key].value
                out.append(outval)
        return out
        
    def free(self):
        unique_sessions = self.get_unique_sessions(self.devices)
        for us in unique_sessions:
            try:
                out = self.tc.root.free(us.key)
                if not out:
                    raise Exception("Unable to free session")
            except Exception as e:
                raise Exception("Unable to free session")
    
    