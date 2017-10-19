#!/usr/bin/python
# coding=utf8

import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import smtplib

mail_host = 'smtp.bqjr.cn'
mail_from = 'chunhui.zhang@bqjr.cn'
mail_pass = 'Zch0302haha'


def addAttch(to_list, subject, content, path):
    msg = MIMEMultipart('related')  ##采用related定义内嵌资源的邮件体
    msgtext = MIMEText(content, _subtype='plain', _charset='utf-8')  ##_subtype有plain,html等格式，避免使用错误

    msg.attach(msgtext)

    os.chdir(path)
    dir = os.getcwd()

    for fn in os.listdir(dir):  ##返回字符串文件名
        print fn
        attach = MIMEText(open(fn, 'rb').read())
        attach["Content-Type"] = 'application/octet-stream'
        attach["Content-Disposition"] = 'attachment; filename=' + fn
        msg.attach(attach)
    msg['Subject'] = subject
    msg['From'] = mail_from
    msg['To'] = to_list
    return msg


def sendMail(msg):
    try:
        server = smtplib.SMTP(mail_host, 25)
        # server = smtplib.SMTP()
        # server.connect(mail_host, 25)
        server.starttls()
        server.login(mail_from, mail_pass)
        server.sendmail(mail_from, msg['To'], msg.as_string())
        server.quit()  ##断开smtp连接
        print "邮件发送成功"
    except Exception, e:
        print "失败" + str(e)


msg = addAttch('chunhui.zhang@bqjr.cn', '标题', '正文', '/home/zch/hqlscripts')
sendMail(msg)
