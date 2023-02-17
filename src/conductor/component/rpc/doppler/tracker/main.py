

from __future__ import absolute_import, print_function
from absl import app
from absl import flags

import uuid
import json
import logging
import hashlib
import rpyc
import socket
import threading
import time
import copy
from rpyc.utils.server import ThreadPoolServer

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
rpyc.core.protocol.DEFAULT_CONFIG['allow_public_attrs'] = True

logger = logging.getLogger("conductor.component.rpc.doppler.tracker")

FLAGS = flags.FLAGS
flags.DEFINE_string("host", "0.0.0.0", "host ip address for server/tracker")
flags.DEFINE_integer("port", 9000, "host port for server/tracker")
flags.DEFINE_integer("port_end", 9199, "host end port for server/tracker")

def hashd(d):
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()
    
class TrackerSession():
    def __init__(self, server, devices):
        self.server = server
        self.devices = copy.deepcopy(devices)
        self.occupied = False
        self.key = hashd(self.get_info())
        self.client_conn = None
        self.client_host = ""
        self.client_port = None
    
    def get_info(self):
        return {
            "host": self.server.host,
            "port": self.server.port,
            "srv_name": self.server.key,
            "devices": list(self.devices),
            "unique_id": str(uuid.uuid4())
        }
    
    def begin(self):
        return self.server.add_session(self.key, self.get_info())
    
    def is_occupied(self):
        return self.occupied
    
    def ping(self):
        if self.client_conn == None:
            try:
                self.client_conn = rpyc.connect(self.client_host, self.client_port)
            except Exception as e:
                return False
        try:
            out = self.client_conn.root.ping()
        except Exception as e:
            out = None
        if out == "pong":
            logger.info("Client keepalive " + self.client_host + ":" + str(self.client_port))
            return True
        else:
            logger.warning("Couldn't ping client " + self.client_host + ":" + str(self.client_port))
            return False

    
    def occupy(self, client_host, client_port):
        if self.occupied:
            logger.error("Session already occupied")
            return False    
        for d in self.devices:
            out = self.server.claim_device(d)
            if not out:
                logger.error("Failed to claim device " + d + " on server " + self.server.key)
                return False
        self.occupied = True
        self.client_host = client_host
        self.client_port = client_port
        return True
    
    def free(self):
        if not self.occupied:
            logger.error("Session already free")
            return False
        for d in self.devices:
            out = self.server.unclaim_device(d)
            if not out:
                logger.error("Failed to unclaim device " + d + " on server " + self.server.key)
                return False
        self.occupied = False
        
        if self.client_conn is not None:
            if not self.client_conn._closed:
                self.client_conn.close()
            self.client_conn = None
        self.client_host = ""
        self.client_port = None
        return True

class Server():
    def __init__(self, host, port, name, devices):
        self.host = host
        self.port = port
        self.key = name
        self.devices = {d: False for d in devices}
        self.conn = rpyc.connect(self.host, self.port)
    
    def get_info(self):
        return {
            "host": self.host,
            "port": self.port,
            "key": self.key,
            "devices": self.devices
        }
    
    def claim_device(self, device):
        if self.devices[device]:
            return False
        self.devices[device] = True
        return True
    
    def unclaim_device(self, device):
        if not self.devices[device]:
            return False
        self.devices[device] = False
        return True
    
    def add_session(self, key, sess_data):
        return self.conn.root.add_session(key, sess_data)
    
    def remove_session(self, key):
        return self.conn.root.remove_session(key)
    
    def ping(self):
        try:
            out = self.conn.root.ping()
        except Exception as e:
            out = None
        if out == "pong":
            logger.info("Server keepalive " + self.key)
            return True
        else:
            logger.warning("Couldn't ping server " + self.key)
            return False
        
class TrackerService(rpyc.Service):
    def __init__(self):
        self.servers = {}
        self.sessions = {}
        self.sessions_by_server = {}
        self.session_threads = {}
        self.server_threads = {}
        # threading.Thread(target=self.server_keepalive, daemon=True).start()
        # threading.Thread(target=self.session_keepalive, daemon=True).start()
        
    def exposed_ping(self):
        return "pong"
    
    def exposed_get_servers(self):
        return {k: s.get_info() for k, s in self.servers.items()}
    
    # client begins a session
    def _begin_session(self, server, devices):
        if server not in self.servers:
            logger.error("Server not registered")
            return None, None
        
        srv = self.servers[server]
        for d in devices:
            if d not in srv.devices:
                logger.error("Device " + d + " not present within the server " + server)
                return None, None
            if srv.devices[d]:
                logger.error("Device " + d + " already claimed on server " + server)
                return None, None
            
        # session selects a server and some devices on it
        session = TrackerSession(srv, devices)
        if session.key in self.sessions:
            logger.warning("Session already exists")
            return session.key, session.get_info()
        
        self.sessions[session.key] = session
        status = self.sessions[session.key].begin()
        if status:
            logger.info("Session " + session.key + " created")

            # add session to server
            if srv.key not in self.sessions_by_server:
                self.sessions_by_server[srv.key] = []
            self.sessions_by_server[srv.key].append(session.key)
            
            return session.key, session.get_info()
        else:
            logger.error("Failed to create session " + session.key)
            return None, None
    
    def exposed_session(self, server, devices, client_host, client_port):
        if server not in self.servers:
            logger.error("Server not registered")
            return None, None
        key, info = self._begin_session(server, devices)
        if key == None:
            logger.error("Failed to begin session")
            return None, None
        else:
            if self.sessions[key].is_occupied():
                logger.error("Session already occupied")
                return None, None
            else:
                self.sessions[key].occupy(client_host, client_port)
                # threading.Thread(target=self.session_keepalive, args=(key,), daemon=True).start()
                return key, info
    
    # client frees a session
    def exposed_free(self, key):
        if key not in self.sessions:
            logger.error("Session not found")
            return False
        else:
            status = self.sessions[key].free()
            if status:
                logger.info("Session " + key + " freed")
                return True
            else:
                logger.error("Failed to free session " + key)
                return False
            
    def session_keepalive(self):
        while True:
            time.sleep(5)
            for key in self.sessions.keys():
                if not self.sessions[key].ping():
                    # session died
                    srv = self.sessions[key].server
                    devs = self.sessions[key].devices
                    for d in devs:
                        srv.unclaim_device(d)
                    srv.remove_session(key)
                    self.sessions_by_server[srv.key].remove(key)
                    del self.sessions[key]
                    logger.info("Removed session " + key)
                    break                
    
    def server_keepalive(self):
        while True:
            time.sleep(5)
            for key in self.servers.keys():
                if not self.servers[key].ping():
                    # server died
                    if key in self.sessions_by_server:
                        for sess in self.sessions_by_server[key]:
                            del self.sessions[sess]
                        del self.sessions_by_server[key]
                        
                    del self.servers[key]
                    logger.info("Removed server " + key)
                    break
    
    # register a server with the tracker
    def exposed_register_server(self, host, port, name, devices):
        if host + ":" + str(port) + "." + name in self.servers:
            logger.error("Server already registered")
            return None
        srv = Server(host, port, name, devices)
        self.servers[srv.key] = srv
        logger.info("Registered server: " + srv.key)
        # threading.Thread(target=self.server_keepalive, args=(srv.key,), daemon=True).start()
        return srv.key

def establish_port(host, port, port_end):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    curp = port
    while curp <= port_end:
        try:
            sock.bind((host, curp))
            sock.close()
            return curp
        except:
            curp += 1
    sock.close()
    logger.error("Unable to find suitable port to bind to")
    return -1

def main(argv):
    del argv
    trckr = TrackerService()
    prt = establish_port(FLAGS.host, FLAGS.port, FLAGS.port_end)
    t = ThreadPoolServer(trckr, hostname=FLAGS.host, port=prt)
    logger.info("Bound tracker to: " + FLAGS.host + ":" + str(prt))
    t.start()

if __name__ == '__main__':
    app.run(main)