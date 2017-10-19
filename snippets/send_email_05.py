#!/usr/bin/env python

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import smtplib

me = "chunhui.zhang@bqjr.cn"
you = "chunhui.zhang@bqjr.cn"

# Create message container - the correct MIME type is multipart/alternative.
msg = MIMEMultipart('alternative')
msg['Subject'] = "Link"
msg['From'] = me
msg['To'] = you

# Create the body of the message (a plain-text and an HTML version).
text = "Hi!\nHow are you?\nHere is the link you wanted:\nhttps://www.python.org"
html = """\
<html>
  <head></head>
  <body>
    <p>Hi!<br>
       How are you?<br>
       Here is the <a href="https://www.python.org">link</a> you wanted.
    </p>
  </body>
</html>
"""

# Record the MIME types of both parts - text/plain and text/html.
part1 = MIMEText(text, 'plain')
part2 = MIMEText(html, 'html')

# Attach parts into message container.
# According to RFC 2046, the last part of a multipart message, in this case
# the HTML message, is best and preferred.
msg.attach(part1)
msg.attach(part2)

host = 'smtp.bqjr.cn'
port = 25
s = smtplib.SMTP(host, port)

sender = 'chunhui.zhang@bqjr.cn'
pwd = 'Zch0302haha'
s.login(sender, pwd)

s.sendmail(me, you, msg.as_string())
s.quit()
