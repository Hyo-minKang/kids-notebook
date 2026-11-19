# MNIST + 합성한 한국식 7(왼쪽 위에 꺾임이 있는 7)로 다시 학습한다.
import sys, numpy as np, base64, json, time
from scipy import ndimage
from sklearn.neural_network import MLPClassifier
d, old_w, out_w, out_ref = sys.argv[1:5]
def idx(p, off): return np.frombuffer(open(p,'rb').read(), dtype=np.uint8, offset=off)
Xtr = idx(d+'/train-images-idx3-ubyte',16).reshape(-1,28,28).astype(np.float32)/255
ytr = idx(d+'/train-labels-idx1-ubyte',8)
Xte = idx(d+'/t10k-images-idx3-ubyte',16).reshape(-1,28,28).astype(np.float32)/255
yte = idx(d+'/t10k-labels-idx1-ubyte',8)
rng = np.random.default_rng(1)

# ---- rec.js 의 rasterize 와 같은 방식으로 획을 28x28 로 그린다 ----
S = 112; YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)
def raster(strokes, R):
    pts = np.concatenate(strokes); x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0; k = (80 - 2*R) / max(w, h, 1e-6)
    ox, oy = (S - w*k)/2 - x0*k, (S - h*k)/2 - y0*k
    buf = np.zeros((S, S), np.float32)
    for st in strokes:
        P = st * k + [ox, oy]
        for (ax, ay), (bx, by) in zip(P[:-1], P[1:]):
            dx, dy = bx-ax, by-ay; l2 = dx*dx + dy*dy
            t = np.clip(((XX-ax)*dx + (YY-ay)*dy) / l2, 0, 1) if l2 > 0 else 0
            dist = np.sqrt((XX-(ax+t*dx))**2 + (YY-(ay+t*dy))**2)
            buf = np.maximum(buf, np.clip(R + 0.5 - dist, 0, 1))
    small = buf.reshape(28, 4, 28, 4).mean((1, 3)); m = small.sum()
    ys, xs = np.mgrid[0:28, 0:28]
    sx, sy = int(round(13.5 - (small*xs).sum()/m)), int(round(13.5 - (small*ys).sum()/m))
    out = np.zeros_like(small)
    src = small[max(0,-sy):28-max(0,sy), max(0,-sx):28-max(0,sx)]
    out[max(0,sy):max(0,sy)+src.shape[0], max(0,sx):max(0,sx)+src.shape[1]] = src
    return out
def seven(hooked):
    w = rng.uniform(0.42, 0.85); ty = rng.uniform(-0.06, 0.10); bx = rng.uniform(0.2, 0.8) * w
    top = [(rng.uniform(-0.04, 0.04), 0.0), (w * 0.5, ty * 0.5 + rng.uniform(-0.03, 0.03)), (w, ty)]
    cx, cy = w * rng.uniform(0.55, 1.0), rng.uniform(0.35, 0.65)          # 내려오는 획의 휘어짐
    ts = np.linspace(0, 1, 9)[1:]
    down = [((1-t)**2*w + 2*(1-t)*t*cx + t*t*bx, (1-t)**2*ty + 2*(1-t)*t*cy + t*t*1.0) for t in ts]
    body = np.array(top + down, np.float32)
    strokes = [body]
    if hooked:
        hook = rng.uniform(0.10, 0.40); hx = rng.uniform(-0.06, 0.06)
        tick = np.array([(hx, hook), (top[0][0], 0.0)], np.float32)
        if rng.random() < 0.3: tick += [rng.uniform(-0.05, 0.02), rng.uniform(-0.03, 0.03)]   # 따로 그어 살짝 떨어진 경우
        if rng.random() < 0.7: strokes = [np.concatenate([tick, body])]
        else: strokes = [tick, body]
    a = np.deg2rad(rng.uniform(-12, 12)); Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    strokes = [(s + rng.normal(0, 0.012, s.shape).astype(np.float32)) @ Rm.T for s in strokes]
    return raster(strokes, rng.uniform(4.0, 6.6))
t0 = time.time()
K_tr = np.stack([seven(True) for _ in range(7000)]); P_tr = np.stack([seven(False) for _ in range(2500)])
K_te = np.stack([seven(True) for _ in range(2000)]); P_te = np.stack([seven(False) for _ in range(1000)])
print('synth done', round(time.time()-t0), flush=True)

def load(p):
    w = json.load(open(p)); L = []
    i = 0
    while 'W%d' % i in w:
        W = np.frombuffer(base64.b64decode(w['W%d'%i]), np.float16).astype(np.float32).reshape(w['s%d'%i])
        b = np.frombuffer(base64.b64decode(w['b%d'%i]), np.float16).astype(np.float32); L.append((W, b)); i += 1
    return L
def pred(L, X):
    a = X.reshape(len(X), -1)
    for i, (W, b) in enumerate(L):
        a = a @ W + b
        if i < len(L)-1: a = np.maximum(a, 0)
    return a.argmax(1)
def report(name, L):
    pk, pp, pm = pred(L, K_te), pred(L, P_te), pred(L, Xte)
    print(name, '| 한국식 7: %.1f%%' % (100*(pk==7).mean()), '오독', dict(zip(*np.unique(pk[pk!=7], return_counts=True))),
          '| 꺾임 없는 7: %.1f%%' % (100*(pp==7).mean()), '| MNIST 시험: %.2f%%' % (100*(pm==yte).mean()),
          '| MNIST 7: %.1f%%' % (100*(pm[yte==7]==7).mean()), '| 다른 숫자를 7로: %d' % ((pm==7)&(yte!=7)).sum(), flush=True)
old = load(old_w); report('이전 모델', old)

def aug(img):
    a = np.deg2rad(rng.uniform(-14,14)); s = rng.uniform(0.82,1.12); sh = rng.uniform(-0.25,0.25)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]) @ np.array([[1,sh],[0,1]]) / s
    c = np.array([13.5,13.5]); t = rng.uniform(-2,2,2)
    out = ndimage.affine_transform(img, R, offset=c - R@c + t, order=1)
    r = rng.random()
    if r < 0.25: out = ndimage.grey_dilation(out, size=(2,2))
    elif r < 0.4: out = ndimage.grey_erosion(out, size=(2,2))
    return np.clip(out,0,1)
A = [Xtr] + [np.stack([aug(x) for x in Xtr]) for _ in range(2)]
syn = np.concatenate([K_tr, P_tr]); syn2 = np.stack([aug(x) for x in syn])
X = np.concatenate(A + [syn, syn2]).reshape(-1,784)
y = np.concatenate([ytr]*3 + [np.full(len(syn)*2, 7, np.uint8)])
print('train set', X.shape, round(time.time()-t0), flush=True)
m = MLPClassifier(hidden_layer_sizes=(256,128), batch_size=256, max_iter=14, learning_rate_init=1e-3, random_state=0, verbose=False)
m.fit(X, y)
w = {}
for i,(W,b) in enumerate(zip(m.coefs_, m.intercepts_)):
    w['W%d'%i] = base64.b64encode(W.astype(np.float16).tobytes()).decode()
    w['b%d'%i] = base64.b64encode(b.astype(np.float16).tobytes()).decode()
    w['s%d'%i] = list(W.shape)
json.dump(w, open(out_w,'w'))
report('새 모델', load(out_w))
ref = {'x': [round(float(v),4) for v in K_te[0].reshape(-1)], 'p': [float(v) for v in m.predict_proba(K_te[:1].reshape(1,-1))[0]], 'y': 7}
json.dump(ref, open(out_ref,'w')); print('time', round(time.time()-t0))
