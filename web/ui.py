#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ui.py -- 本机网页控制台：点按钮跑 mlogpt/gen.py 的构建与（可选的）模拟。

    python web/ui.py                 # http://127.0.0.1:8771/
    python web/ui.py --work out/     # 换产物目录
    python web/ui.py --port 9000 --no-open

构建是纯 numpy，只要权重就能跑；--run 那条路（游戏内 144 台分词器）需要
外部 Mindustry v160.7 jar + JRE21 + 姊妹仓 mlog-agentskill 的模拟器，见
docs/REPRODUCE.md。没有模拟器时构建照常可用，Run 按钮会失败并打印原因。

构建产物可以直接「安装到游戏蓝图目录」（Windows: %APPDATA%/Mindustry/schematics，
其余平台: ~/.local/share/Mindustry/schematics）：页面会显示这个目录，安装只拷贝
文件、不启动游戏。GET /api/env 报目录，POST /api/install 拷蓝图（重名覆盖）。
"""
import json, os, re, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))              # web/
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))          # 仓库根
GEN = os.path.join(ROOT, 'mlogpt', 'gen.py')
WORK = os.path.join(ROOT, 'out')                               # 产物目录，--work 可改
JOBS = {}
LOCK = threading.Lock()
SEQ = [0]


def _ls(pattern, base=None):
    """列 base 下的文件；返回的名字能直接回传给 /api/build（见 _resolve）。"""
    out = []
    base = base or WORK
    try:
        for fn in sorted(os.listdir(base)):
            if re.search(pattern, fn) and os.path.isfile(os.path.join(base, fn)):
                out.append(fn)
    except OSError:
        pass
    return out


def _resolve(name):
    """把 UI 传来的文件名解析成绝对路径：先 WORK，再仓库根。"""
    if not name:
        return ''
    if os.path.isabs(name):
        return name
    for base in (WORK, ROOT):
        cand = os.path.join(base, name)
        if os.path.exists(cand):
            return cand
    return os.path.join(WORK, name)      # 交给 gen.py 报「找不到」


def _worker(job):
    args = job['args']
    cmd = [sys.executable, '-X', 'utf8', GEN] + args
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    job['cmd'] = ' '.join(cmd)
    try:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=1,
                             universal_newlines=True, encoding='utf-8', errors='replace')
        for line in p.stdout:
            line = line.rstrip()
            if not line:
                continue
            with LOCK:
                job['lines'].append(line)
                if len(job['lines']) > 6000:
                    del job['lines'][:2000]
            m = re.search(r'tick (\d+)', line)
            if m:
                job['tick'] = int(m.group(1))
        p.wait()
        job['rc'] = p.returncode
    except Exception as exc:
        job['rc'] = -1
        with LOCK:
            job['lines'].append('启动失败: %r' % (exc,))
    job['t1'] = time.time()
    jp = job.get('json_path')
    if jp and os.path.exists(jp):
        try:
            with open(jp, 'r', encoding='utf-8') as f:
                job['result'] = json.load(f)
        except Exception as exc:
            with LOCK:
                job['lines'].append('读取结果失败: %r' % (exc,))
    job['status'] = 'done' if job['rc'] == 0 else 'failed'


def start_job(kind, args, meta):
    with LOCK:
        SEQ[0] += 1
        jid = '%s-%d' % (kind, SEQ[0])
        job = dict(id=jid, kind=kind, args=args, meta=meta, status='running',
                   lines=[], t0=time.time(), t1=None, rc=None, result=None,
                   json_path=meta.get('json_path'), tick=0)
        JOBS[jid] = job
    threading.Thread(target=_worker, args=(job,), daemon=True).start()
    return job


def _flag(p, key):
    return str(p.get(key, '')).lower() in ('1', 'true', 'on', 'yes')


def build_args(p):
    d = int(p.get('d', 128)); S = int(p.get('S', 16)); L = int(p.get('L', 4))
    V = int(p.get('V', 4096)); N = int(p.get('N', 8)); cap = int(p.get('cap', 900))
    # 产物文件名：UI 里填的是「名字」，不带目录；缺 .msch 自动补，重名直接覆盖
    name = (p.get('out') or '').strip() or ('mlogpt_d%ds%dl%dv%d.msch' % (d, S, L, V))
    if not name.lower().endswith('.msch'):
        name += '.msch'
    name = os.path.basename(name)
    out = os.path.join(WORK, name)
    stats = p.get('stats') or (out + '.json')
    a = ['--build',
         '--d', str(d), '--S', str(S), '--L', str(L), '--V', str(V), '--N', str(N),
         '--cap', str(cap), '--seed-tok', str(int(p.get('seed_tok', 0))),
         '--seed', str(int(p.get('seed', 7))),
         '--out', out, '--stats', stats]
    if _flag(p, 'table_free'):
        a.append('--table-free')
    w = _resolve(p.get('weights') or '')
    if w:
        a += ['--weights', w]
    v = _resolve(p.get('vocab') or '')
    if v:
        a += ['--vocab', v]
    if _flag(p, 'input'):
        a.append('--input')
        txt = p.get('input_text') or ''
        if txt:
            a += ['--input-text', txt]
    return a, dict(msch=out, stats=stats, json_path=stats)


def run_args(p):
    path = _resolve(p.get('msch') or '')
    rj = path + '.run.json'
    a = ['--run', path, '--stats', path + '.json',
         '--slice', str(int(p.get('slice', 1))),
         '--max-ticks', str(int(p.get('ticks', 45))),
         '--extra', str(int(p.get('extra', 8))), '--json', rj]
    txt = p.get('input_text')
    if txt:
        a += ['--input-text', txt]
    txt2 = p.get('input_text2')
    if txt2:
        a += ['--input-text2', txt2]
    return a, dict(msch=path, json_path=rj)


def html_page():
    """每次现读 ui.html：改前端刷新页面就生效（不用重启服务）。"""
    return open(os.path.join(HERE, 'ui.html'), 'r', encoding='utf-8').read()


def schematic_dir():
    """Mindustry 的蓝图目录（Windows: %APPDATA%/Mindustry/schematics）。"""
    if os.name == 'nt':
        base = os.environ.get('APPDATA') or os.path.expanduser(r'~\AppData\Roaming')
        return os.path.join(base, 'Mindustry', 'schematics')
    return os.path.expanduser('~/.local/share/Mindustry/schematics')


def _files():
    """权重/词表既可能放在仓库 weights/，也可能落在产物目录里。"""
    weights, vocab = [], []
    for base, prefix in ((os.path.join(ROOT, 'weights'), 'weights/'), (WORK, '')):
        weights += [prefix + n for n in _ls(r'.*\.npz$', base)]
        vocab += [prefix + n for n in _ls(r'.*\.json$', base)]
    return dict(weights=weights, vocab=vocab, msch=_ls(r'\.msch$'))


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ('/', '/index.html'):
            self._send(200, html_page(), 'text/html; charset=utf-8')
            return
        if u.path == '/api/env':
            sd = schematic_dir()
            self._send(200, json.dumps(dict(schematics=sd, exists=os.path.isdir(sd),
                                            work=WORK), ensure_ascii=False))
            return
        if u.path == '/api/files':
            self._send(200, json.dumps(_files()))
            return
        if u.path == '/api/job':
            q = parse_qs(u.query)
            jid = (q.get('id') or [''])[0]
            frm = int((q.get('from') or ['0'])[0])
            job = JOBS.get(jid)
            if not job:
                self._send(404, json.dumps(dict(error='no such job')))
                return
            with LOCK:
                lines = job['lines'][frm:]
                total = len(job['lines'])
            d = dict(id=jid, kind=job['kind'], status=job['status'], rc=job['rc'],
                     elapsed=round((job['t1'] or time.time()) - job['t0'], 1),
                     lines=lines, next=total, tick=job.get('tick', 0),
                     meta=job['meta'], result=job['result'], cmd=job.get('cmd'))
            self._send(200, json.dumps(d, ensure_ascii=False))
            return
        self._send(404, json.dumps(dict(error='not found')))

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get('Content-Length') or 0)
        try:
            p = json.loads(self.rfile.read(n).decode('utf-8') or '{}')
        except Exception:
            p = {}
        if u.path == '/api/build':
            a, meta = build_args(p)
            job = start_job('build', a, meta)
            self._send(200, json.dumps(dict(id=job['id'])))
            return
        if u.path == '/api/install':
            # 把构建产物拷进游戏蓝图目录（重名覆盖）：先写 .part 再原子替换，
            # 免得游戏正开着蓝图列表时读到半个文件
            name = os.path.basename((p.get('msch') or '').strip())
            src = os.path.join(WORK, name) if name else ''
            if not name or not os.path.isfile(src):
                self._send(404, json.dumps(dict(error='找不到蓝图: %r' % (p.get('msch'),)),
                                           ensure_ascii=False))
                return
            sd = schematic_dir()
            os.makedirs(sd, exist_ok=True)
            dst = os.path.join(sd, name)
            replaced = os.path.isfile(dst)
            tmp = dst + '.part'
            with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
            os.replace(tmp, dst)
            self._send(200, json.dumps(dict(ok=True, dst=dst, bytes=os.path.getsize(dst),
                                            src=src, replaced=replaced), ensure_ascii=False))
            return
        if u.path == '/api/run':
            a, meta = run_args(p)
            job = start_job('run', a, meta)
            self._send(200, json.dumps(dict(id=job['id'])))
            return
        self._send(404, json.dumps(dict(error='not found')))


def main():
    global WORK
    port = 8771
    noopen = False
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--port' and i + 1 < len(argv):
            port = int(argv[i + 1]); i += 2; continue
        if a == '--work' and i + 1 < len(argv):
            WORK = os.path.abspath(argv[i + 1]); i += 2; continue
        if a == '--no-open':
            noopen = True
        i += 1
    os.makedirs(WORK, exist_ok=True)
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    url = 'http://127.0.0.1:%d/' % port
    print('mlogpt UI:', url)
    print('  gen   :', GEN)
    print('  产物  :', WORK)
    print('关闭：Ctrl+C')
    if not noopen:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
