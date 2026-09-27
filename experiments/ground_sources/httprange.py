"""Seekable read-only file object over HTTP Range requests, with a byte counter and tally logging."""
import io, urllib.request, time, os
TALLY = '/home/thinklex/Dropbox/2025_phd/2026/35_PRC_Data_Challenge_OpenSky/00_src/prc2026/cache/ground_sources/tally.tsv'
class HTTPRange(io.RawIOBase):
    def __init__(self, url, size=None):
        self.url, self.pos, self.nbytes = url, 0, 0
        if size is None:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=60) as r: size = int(r.headers['Content-Length'])
        self.size = size
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b''
        for i in range(5):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'})
                with urllib.request.urlopen(req, timeout=300) as r: b = r.read()
                break
            except Exception as e:
                print('retry', i, e); time.sleep(10 * (i + 1))
        else: raise IOError('range read failed')
        self.pos += len(b); self.nbytes += len(b)
        return b
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
    def log(self, note=''):
        with open(TALLY, 'a') as f: f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{self.nbytes}\t{self.url} (range reads{': ' + note if note else ''})\n")
