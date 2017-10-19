#coding:utf-8
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import smtplib

#创建一个带附件的实例
msg = MIMEMultipart()

#构造附件1
att1 = MIMEText(open('/home/zch/hqlscripts/noproblem.sql', 'rb').read(), 'base64', 'gb2312')
att1["Content-Type"] = 'application/octet-stream'
att1["Content-Disposition"] = 'attachment; filename="noproblem.sql"'
msg.attach(att1)

#构造附件2
att2 = MIMEText(open('/home/zch/hqlscripts/summary0717.sql', 'rb').read(), 'base64', 'gb2312')
att2["Content-Type"] = 'application/octet-stream'
att2["Content-Disposition"] = 'attachment; filename="summary0717.sql"'
msg.attach(att2)

#加邮件头
msg['to'] = 'chunhui.zhang@bqjr.cn'
msg['from'] = 'chunhui.zhang@bqjr.cn'
msg['subject'] = 'hello world with attachment'
#发送邮件
try:
host = 'smtp.bqjr.cn'
port = 25
s = smtplib.SMTP(host, port)
s.login('chunhui.zhang@bqjr.cn','Zch0302haha')
s.sendmail(msg['from'], msg['to'],msg.as_string())
s.quit()
print '发送成功'
except Exception, e:
    print str(e)