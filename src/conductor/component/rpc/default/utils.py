
import socket
import errno
import struct

# magic header for RPC tracker(control plane)
RPC_TRACKER_MAGIC = 0x2F271

def recvall(sock, nbytes):
    """Receive all nbytes from socket.

    Parameters
    ----------
    sock: Socket
       The socket

    nbytes : int
       Number of bytes to be received.
    """
    res = []
    nread = 0
    while nread < nbytes:
        chunk = sock.recv(min(nbytes - nread, 1024))
        if not chunk:
            raise IOError("connection reset")
        nread += len(chunk)
        res.append(chunk)
    return b"".join(res)

def get_addr_family(addr):
    res = socket.getaddrinfo(addr[0], addr[1], 0, 0, socket.IPPROTO_TCP)
    return res[0][0]

def conn_addr(addr):
    try:
        sock = socket.socket(get_addr_family(addr), socket.SOCK_STREAM)
        sock.connect(addr)
        return sock
    except socket.error as sock_err:
        if sock_err.args[0] not in (errno.ECONNREFUSED,):
            raise sock_err
        return None

def connect(addr):
    sock = conn_addr(addr)
    if sock is None:
        return False
    sock.sendall(struct.pack("<i", RPC_TRACKER_MAGIC))
    magic = struct.unpack("<i", recvall(sock, 4))[0]
    if magic != RPC_TRACKER_MAGIC:
        return False
    else:
        return True

def find_tracker_port(host, port, port_end):
    cur_port = port
    while cur_port <= port_end:
        retval = connect((host, cur_port))        
        if retval:
            return cur_port
        else:
            cur_port += 1
            continue