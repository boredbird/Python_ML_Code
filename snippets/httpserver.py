#!/usr/bin/python
# coding=utf-8

from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler

import time

starttime = time.time()


class MyHandler(BaseHTTPRequestHandler):
    #    '''Definition of the request handler.'''
    def _writeheaders(self, doc):

        if doc is None:
            self.send_response(404)
        else:
            self.send_response(200)

        self.send_header("Content-type", "text/html")
        self.end_headers()

    def _getdoc(self, filename):
        '''Handle a request for a document,returning one of two different page as as appropriate.'''
        if filename == '/':
            return '''
                <html>
                    <head>
                        <title>Samlle Page</title>
                        <script type="text/javascript">
                            //alert("hello");
                        </script>
                    </head>

                    <body>
                        This is a sample page.You can also look at the
                        <a href="stats.html">Server statistics</a>.
                    </body>
                </html>
                '''
        elif filename == '/stats.html':
            return '''
                <html>
                    <head>
                        <title>Statistics</title>
                    </head>

                    <body>
                        this server has been running for %d seconds.
                    </body>
                </html>
                ''' % int(time.time() - starttime)
        else:
            return None

    def do_HEAD(self):
        '''Handle a request for headers only'''
        doc = self._getdoc(self.path)
        self._writeheaders(doc)

    def do_GET(self):
        '''Handle a request for headers and body'''
        print "Get path is:%s" % self.path
        doc = self._getdoc(self.path)
        self._writeheaders(doc)
        if doc is None:
            self.wfile.write('''
                                <html>
                                    <head>
                                        <title>Not Found</title>
                                        <body>
                                            The requested document '%s' was not found.
                                        </body>
                                    </head>
                                </html>
                                ''' % (self.path))

        else:
            self.wfile.write(doc)


# Create the pbject and server requests
serveaddr = ('', 8000)
httpd = HTTPServer(serveaddr, MyHandler)
print "Base serve is start add is %s port is %d" % (serveaddr[0], serveaddr[1])
httpd.serve_forever()