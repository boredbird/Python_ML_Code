#coding=utf8
from multiprocessing import Pool

def do_add(n1, n2):
    return n1+n2

if __name__ == "__main__":
    pool = Pool(5)
    print pool.map(do_add, [(1,2),(3,4),(5,6)])
    pool.close()
    pool.join()