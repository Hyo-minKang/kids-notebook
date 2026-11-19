# MNIST + 합성한 각진 5 / 각진 9 로 다시 학습한다. (5를 9로 읽는 문제)
import sys, numpy as np, base64, json, time
from scipy import ndimage
from sklearn.neural_network import MLPClassifier
src = open('train2.py', encoding='utf-8').read()
sys.argv = ['x', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
exec(src.split("def seven")[0])                                     # 데이터 읽기, raster()
exec("def load" + src.split("def load")[1].split("def report")[0])  # load(), pred()
exec("def aug" + src.split("def aug")[1].split("A = [Xtr]")[0])     # aug()
rng = np.random.default_rng(5)
def chaikin(P, n):
    for _ in range(n):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]): Q += [0.75*a + 0.25*b, 0.25*a + 0.75*b]
        P = np.array(Q + [P[-1]], np.float32)
    return P
def finish(strokes):
    a = np.deg2rad(rng.uniform(-10, 10)); Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    return raster([(s + rng.normal(0, 0.01, s.shape).astype(np.float32)) @ Rm.T for s in strokes], rng.uniform(4.0, 6.6))
def five():
    w = rng.uniform(0.42, 0.8); hb = rng.uniform(0.22, 0.5); xs = rng.uniform(-0.03, 0.06)
    belly = np.array([(xs, rng.uniform(0, 0.06)), (xs + rng.uniform(-0.05, 0.05), hb), (w, hb + rng.uniform(-0.04, 0.06)),
                      (w + rng.uniform(-0.04, 0.04), 1 - rng.uniform(0.08, 0.22)), (w * rng.uniform(0.4, 0.7), 1.0),
                      (w * rng.uniform(0.0, 0.3), 1 - rng.uniform(0, 0.1))], np.float32)
    belly = chaikin(belly, int(rng.integers(0, 3)))
    bar = np.array([(xs + rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.04)), (w * rng.uniform(0.8, 1.15), rng.uniform(-0.07, 0.05))], np.float32)
    return finish([belly, bar])
def nine():
    w = rng.uniform(0.42, 0.8); hb = rng.uniform(0.3, 0.55)
    box = np.array([(w, rng.uniform(0, 0.05)), (rng.uniform(0, 0.08), rng.uniform(-0.02, 0.04)), (rng.uniform(-0.03, 0.05), hb), (w, hb + rng.uniform(-0.05, 0.03))], np.float32)
    tail = [(w + rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.05)), (w + rng.uniform(-0.04, 0.04), 1 - rng.uniform(0.05, 0.2))]
    if rng.random() < 0.6: tail += [(w * rng.uniform(0.4, 0.7), 1.0), (w * rng.uniform(0.0, 0.3), 1 - rng.uniform(0, 0.1))]
    else: tail[-1] = (tail[-1][0] + rng.uniform(-0.15, 0.05), 1.0)
    n = int(rng.integers(0, 3))
    return finish([chaikin(box, n), chaikin(np.array(tail, np.float32), n)])
def squat5():   # 화면의 5 처럼 윗칸이 낮고 넓으며, 윗줄이 오른쪽 끝까지 닿는 5
    w = rng.uniform(0.45, 0.65); hb = rng.uniform(0.2, 0.36); xs = rng.uniform(-0.02, 0.04)
    belly = np.array([(xs, rng.uniform(-0.06, 0.02)), (xs + rng.uniform(-0.04, 0.03), hb), (w * rng.uniform(0.85, 1.0), hb + rng.uniform(-0.06, 0.02)),
                      (w + rng.uniform(-0.02, 0.04), hb + rng.uniform(0.04, 0.12)), (w + rng.uniform(-0.03, 0.04), 1 - rng.uniform(0.1, 0.2)),
                      (w * rng.uniform(0.6, 0.85), 1 - rng.uniform(0, 0.05)), (w * rng.uniform(0.35, 0.55), 1.0)], np.float32)
    bar = np.array([(xs + rng.uniform(-0.02, 0.03), rng.uniform(0.0, 0.07)), (w * rng.uniform(0.95, 1.1), rng.uniform(0.0, 0.07))], np.float32)
    return finish([chaikin(belly, int(rng.integers(0, 2))), bar])
t0 = time.time()
F_tr = np.stack([five() for _ in range(6000)] + [squat5() for _ in range(6000)]); N_tr = np.stack([nine() for _ in range(5000)])
F_te = np.stack([five() for _ in range(2000)]); N_te = np.stack([nine() for _ in range(2000)])
# 화면에 찍힌 5 를 그대로 옮긴 것
shot = raster([np.array([(570,400),(635,400)], np.float32), np.array([(570,392),(568,440),(628,438),(636,450),(636,500),(625,515),(600,520)], np.float32)], 5.2)
def report(name, L):
    pf, pn, pm = pred(L, F_te), pred(L, N_te), pred(L, Xte)
    print(name, '| 화면의 5 ->', pred(L, shot[None])[0], '| 각진 5: %.1f%%' % (100*(pf==5).mean()), '오독', dict(zip(*np.unique(pf[pf!=5], return_counts=True))),
          '| 각진 9: %.1f%%' % (100*(pn==9).mean()), '| MNIST 전체 %.2f%%' % (100*(pm==yte).mean()),
          '5: %.1f%%' % (100*(pm[yte==5]==5).mean()), '9: %.1f%%' % (100*(pm[yte==9]==9).mean()), flush=True)
report('이전 모델', load(old_w))
A = [Xtr] + [np.stack([aug(x) for x in Xtr]) for _ in range(2)]
syn = np.concatenate([F_tr, N_tr]); ysyn = np.concatenate([np.full(len(F_tr), 5, np.uint8), np.full(len(N_tr), 9, np.uint8)])
X = np.concatenate(A + [syn, np.stack([aug(x) for x in syn])]).reshape(-1, 784)
y = np.concatenate([ytr]*3 + [ysyn, ysyn])
m = MLPClassifier(hidden_layer_sizes=(256,128), batch_size=256, max_iter=14, learning_rate_init=1e-3, random_state=0, verbose=False)
m.fit(X, y)
w = {}
for i,(W,b) in enumerate(zip(m.coefs_, m.intercepts_)):
    w['W%d'%i] = base64.b64encode(W.astype(np.float16).tobytes()).decode()
    w['b%d'%i] = base64.b64encode(b.astype(np.float16).tobytes()).decode()
    w['s%d'%i] = list(W.shape)
json.dump(w, open(out_w,'w'))
report('새 모델', load(out_w)); print('time', round(time.time()-t0))
