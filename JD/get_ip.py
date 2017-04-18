#!/usr/bin/env python

import socket

def getip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('www.baidu.com', 0))
        ip=s.getsockname()[0]
    except:
        ip=""
    finally:
        s.close()
    return ip

if __name__=='__main__':
    print getip()