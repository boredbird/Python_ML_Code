# -*- coding: cp936 -*-
import os
import time

def mkSubFile(lines,head,srcName,sub):
    [des_filename, extname] = os.path.splitext(srcName)
    filename  = des_filename + '_' + str(sub) + extname
    # print( 'make file: %s' %filename)
    print('%s\tGEN SUB FILE: \t%s' %(time.asctime(time.localtime(time.time())),filename))
    fout = open(filename,'w')
    try:
        fout.writelines([head])
        fout.writelines(lines)
        return sub + 1
    finally:
        fout.close()

def splitByLineCount(filename,count):
    fin = open(filename,'r')
    try:
        head = fin.readline()
        buf = []
        sub = 1
        for line in fin:
            buf.append(line)
            if len(buf) == count:
                sub = mkSubFile(buf,head,filename,sub)
                buf = []
        if len(buf) != 0:
            sub = mkSubFile(buf,head,filename,sub)
    finally:
        fin.close()


if __name__ == "__main__":
    dataset_path = 'E:\\ScoreCard\\cs_model\\cs_m1_pos_model_daily\\raw_data\\' + 'm1_rsx_cs_unify_model_features_201705_daily.csv'
    print time.asctime(time.localtime(time.time())),'\t BEGIN'
    splitByLineCount(dataset_path,500000)
    print time.asctime(time.localtime(time.time())),'\t END'

