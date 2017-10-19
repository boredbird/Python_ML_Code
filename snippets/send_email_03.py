#coding:utf-8
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import smtplib

host = 'smtp.bqjr.cn'
port = 25
sender = 'chunhui.zhang@bqjr.cn'
pwd = 'Zch0302haha'
receiver = 'chunhui.zhang@bqjr.cn'


msg = MIMEMultipart()
msg['subject'] = 'Hello world'
msg['from'] = sender
msg['to'] = receiver

att1 = MIMEText(open('/home/zch/hqlscripts/noproblem.sql', 'rb').read(), 'base64', 'utf-8')
att1["Content-Type"] = 'application/octet-stream'
att1["Content-Disposition"] = 'attachment; filename="noproblem.sql"'
msg.attach(att1)

s = smtplib.SMTP(host, port)
s.login(sender, pwd)
s.sendmail(sender, receiver, msg.as_string())

