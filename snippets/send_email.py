# -*- coding:utf-8 -*-
__author__ = 'chunhui.zhang'

from email.mime.text import MIMEText

import smtplib

mailto_list = ['chunhui.zhang@bqjr.cn',]
# mail_host = 'smtp.bqjr.cn'  # 设置服务器
mail_host = 'smtp.exmail.qq.com'  # 设置服务器
mail_user = "chunhui.zhang"  # 用户名
mail_pass = "thinkabout1"  # 口令
mail_postfix = "bqjr.cn"  # 发件箱的后缀

server = smtplib.SMTP()
server.connect(mail_host)
server.login(mail_user, mail_pass)
server.sendmail(me, to_list, msg.as_string())
server.close()



def send_mail(to_list, sub, content):
    me = "hello" + "<" + mail_user + "@" + mail_postfix + ">"
    msg = MIMEText(content, _subtype='plain', _charset='gb2312')
    msg['Subject'] = sub
    msg['From'] = me
    msg['To'] = ";".join(to_list)
    try:
        server = smtplib.SMTP()
        server.connect(mail_host)
        server.login(mail_user, mail_pass)
        server.sendmail(me, to_list, msg.as_string())
        server.close()
        return True
    except Exception, e:
        print str(e)
        return False


if __name__ == '__main__':
    if send_mail(mailto_list, "hello", "hello world！"):
        print "发送成功"
    else:
        print "发送失败"