import multiprocessing
# import sys
# sys.path.append(r'E:\Code\Python_ML_Code\location')

def worker(num):
    """thread worker function"""
    print 'Worker:', num
    return


if __name__ == "__main__":
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,))
        jobs.append(p)
        p.start()