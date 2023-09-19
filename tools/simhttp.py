__version__ = "0.1"
__all__ = ["SimpleHTTPRequestHandler"]

import html
import http.server
import mimetypes
import os
import posixpath
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO

import json


def scale_convert(num):
    if num > 1e9:
        res = num / 1e9
        unit = 'G'
    else:
        if num > 1e6:
            res = num / 1e6
            unit = 'M'
        else:
            res = num / 1e3
            unit = 'K'
    return '{:3.1f}{}'.format(res, unit).rjust(6, ' ')


class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """a simple http file server, surport upload\download\delete
    """

    server_version = "SimpleHTTPWithUpload/" + __version__

    def do_GET(self):
        f = self.send_head()
        if f:
            self.copyfile(f, self.wfile)
            f.close()

    def do_HEAD(self):
        f = self.send_head()
        if f:
            f.close()

    def do_POST(self):
        r, info = self.deal_post_data()
        if info.startswith('Delete'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"message": "ok"}).encode('utf-8'))
        else:
            f = BytesIO()
            f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
            f.write(b"<html>\n<title>Upload Result Page</title>\n")
            f.write(b"<body>\n<h2>Upload Result Page</h2>\n")
            f.write(b"<hr>\n")
            if r:
                f.write(b"<strong>Success:</strong>")
            else:
                f.write(b"<strong>Failed:</strong>")
            f.write(info.encode())
            f.write(("<br><a href=\"%s\">back</a>" % self.headers['referer']).encode())
            f.write(b"</body>\n</html>\n")
            length = f.tell()
            f.seek(0)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if f:
                self.copyfile(f, self.wfile)
                f.close()

    def deal_post_data(self):
        content_type = self.headers['content-type']
        if not content_type:
            return (False, "Content-Type header doesn't contain boundary")
        if content_type == 'application/json':
            length = int(self.headers.get('content-length'))
            message = json.loads(self.rfile.read(length).decode('utf-8'))
            for _path in message['pathStr'].strip(';').split(';'):
                if os.path.exists(_path):
                    os.remove(_path)
            return (True, "Delete;%s" % message['pathStr'])
        else:
            boundary = content_type.split("=")[1].encode()
            remainbytes = int(self.headers['content-length'])
            line = self.rfile.readline()
            remainbytes -= len(line)
            if not boundary in line:
                return (False, "Content NOT begin with boundary")
            line = self.rfile.readline()
            remainbytes -= len(line)
            fn = re.findall(r'Content-Disposition.*name="file"; filename="(.*)"', line.decode())
            if not fn:
                return (False, "Can't find out file name...")
            path = self.translate_path(self.path)
            fn = os.path.join(path, fn[0])
            line = self.rfile.readline()
            remainbytes -= len(line)
            line = self.rfile.readline()
            remainbytes -= len(line)
            try:
                out = open(fn, 'wb')
            except IOError:
                return (False, "Can't create file to write, do you have permission to write?")

            preline = self.rfile.readline()
            remainbytes -= len(preline)
            while remainbytes > 0:
                line = self.rfile.readline()
                remainbytes -= len(line)
                if boundary in line:
                    preline = preline[0:-1]
                    if preline.endswith(b'\r'):
                        preline = preline[0:-1]
                    out.write(preline)
                    out.close()
                    return (True, "File '%s' upload success!" % fn)
                else:
                    out.write(preline)
                    preline = line
            return (False, "Unexpect Ends of data.")

    def send_head(self):
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # redirect browser - doing basically what apache does
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        ctype = self.guess_type(path)
        try:
            # Always read in binary mode. Opening files in text mode may cause
            # newline translations, making the actual size of the content
            # transmitted *less* than the content-length!
            f = open(path, 'rb')
        except IOError:
            self.send_error(404, "File not found")
            return None
        self.send_response(200)
        self.send_header("Content-type", ctype)
        fs = os.fstat(f.fileno())
        self.send_header("Content-Length", str(fs[6]))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()
        return f

    def list_directory(self, path):
        js_script = """
function delfile(path){
    console.log(path)
    var files=document.getElementsByName("cb");
    var pathStr=""
    for(var x=0;x<files.length;x++){
        if(files[x].checked){
            var fn=files[x].value
            var curPath=path+"/"+fn+";"
            console.log(curPath)
            pathStr+=curPath
        }
    }
    $.ajax
    ('%s', {
        type: "POST",
        dataType: 'json',
        contentType: 'application/json',
        async: false,
        data: '{"pathStr": "' + pathStr + '"}',
        success: function () {
            console.log("delete");
            window.location.href = "/";
        }
    })
}\n"""
        self.js_script = js_script % self.headers['referer']
        try:
            list = os.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None
        list.sort(key=lambda a: a.lower())
        f = BytesIO()
        displaypath = html.escape(urllib.parse.unquote(self.path))
        f.write(b'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">')
        f.write(("<html>\n<title>Directory listing for %s</title>\n" % displaypath).encode())
        f.write(('<script src="http://10.255.85.19:8902/lijianwei/envs/jquery-3.6.0.min.js"></script>\n').encode())
        f.write(("<script>%s</script>\n" % self.js_script).encode())
        f.write(("<body>\n<h2>Directory listing for %s</h2>\n" % displaypath).encode())
        f.write(b"<hr>\n")
        f.write(b"<form ENCTYPE=\"multipart/form-data\" method=\"post\">")
        f.write(b"<input name=\"file\" type=\"file\"/>")
        f.write(b"<input type=\"submit\" value=\"upload\"/>&nbsp;&nbsp;&nbsp;")
        f.write(('<input type="button" name="btn" onclick="delfile(%s)" value="delete"/>\n' % ("\'"+path+"\'")).encode())
        f.write(b"<hr>\n<ul>\n")

        dir_list = []
        file_list = []
        dot_list = []
        for name in list:
            fullname = os.path.join(path, name)
            if os.path.isdir(fullname):
                if name.startswith('.'):
                    dot_list.append(name)
                else:
                    dir_list.append(name)
            else:
                if name.startswith('.'):
                    dot_list.append(name)
                else:
                    file_list.append(name)
        list = dir_list + file_list + dot_list

        for name in list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            # Append / for directories or @ for symbolic links
            if os.path.isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"
                f.write(('<li><a href="%s">%s</a>\n'
                     % (urllib.parse.quote(linkname), html.escape(displayname))).encode())
                continue
            if os.path.islink(fullname):
                displayname = name + "@"
                # Note: a link to a directory displays with @ and links with /
            file_size = scale_convert(os.path.getsize(fullname))
            f.write(
                (
                    '<li><input type="checkbox" name="cb" value="%s"/> <a href="%s">%s</a> %s\n'
                    % (html.escape(displayname), urllib.parse.quote(linkname), html.escape(displayname), file_size)
                ).encode()
            )
            
        f.write(b"</ul>\n<hr>\n</body>\n</html>\n")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

    def translate_path(self, path):
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        path = posixpath.normpath(urllib.parse.unquote(path))
        words = path.split('/')
        words = [_f for _f in words if _f]
        path = os.getcwd()
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir): continue
            path = os.path.join(path, word)
        return path

    def copyfile(self, source, outputfile):
        shutil.copyfileobj(source, outputfile)

    def guess_type(self, path):
        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']

    if not mimetypes.inited:
        mimetypes.init()  # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream',  # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
    })

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', '-b', default='', metavar='ADDRESS',
                        help='Specify alternate bind address '
                             '[default: all interfaces]')
    parser.add_argument('--port', '-p', default=8900, type=int,
                        help='Specify alternate port [default: 8900]')
    args = parser.parse_args()
    http.server.test(HandlerClass=SimpleHTTPRequestHandler, port=args.port, bind=args.bind) 