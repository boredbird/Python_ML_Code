# -*- coding:utf-8 -*-
__author__ = 'maomaochong'
def memory_usage_psutil():
    # return the memory usage in MB
    import psutil,os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def get_current_obj(a=[]):
    a.append([0]*1000)
    return a

@profile
def main():
    obj = []
    for i in range(10000):
        obj = get_current_obj(obj)
        if(i%100==0):
            print(memory_usage_psutil())


# 运行：E:\Code\Python_ML_Code\cs_model>python -m memory_profiler test_memory_profiler.py

if __name__=='__main__':
    main()