import sys, numpy as np, base64, json, time
from scipy import ndimage
from sklearn.neural_network import MLPClassifier
d = sys.argv[1]
def idx(p, off): return np.frombuffer(open(p,'rb').read(), dtype=np.uint8, offset=off)
Xtr = idx(d+'/train-images-idx3-ubyte',16).reshape(-1,28,28).astype(np.float32)/255
ytr = idx(d+'/train-labels-idx1-ubyte',8)
Xte = idx(d+'/t10k-images-idx3-ubyte',16).reshape(-1,28,28).astype(np.float32)/255
yte = idx(d+'/t10k-labels-idx1-ubyte',8)
rng = np.random.default_rng(0)
def aug(img):
    a = np.deg2rad(rng.uniform(-14,14)); s = rng.uniform(0.82,1.12); sh = rng.uniform(-0.25,0.25)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]) @ np.array([[1,sh],[0,1]]) / s
    c = np.array([13.5,13.5]); t = rng.uniform(-2,2,2)
    out = ndimage.affine_transform(img, R, offset=c - R@c + t, order=1)
    r = rng.random()
    if r < 0.25: out = ndimage.grey_dilation(out, size=(2,2))
    elif r < 0.4: out = ndimage.grey_erosion(out, size=(2,2))
    return np.clip(out,0,1)
t0=time.time()
A = [Xtr] + [np.stack([aug(x) for x in Xtr]) for _ in range(2)]
X = np.concatenate(A).reshape(-1,784); y = np.concatenate([ytr]*3)
print('aug done', time.time()-t0, X.shape, flush=True)
m = MLPClassifier(hidden_layer_sizes=(256,128), batch_size=256, max_iter=14, learning_rate_init=1e-3, random_state=0, verbose=True)
m.fit(X,y)
print('test acc', m.score(Xte.reshape(-1,784), yte), 'time', time.time()-t0, flush=True)
Xa = np.stack([aug(x) for x in Xte]).reshape(-1,784)
print('aug test acc', m.score(Xa, yte))
w = {}
for i,(W,b) in enumerate(zip(m.coefs_, m.intercepts_)):
    w['W%d'%i] = base64.b64encode(W.astype(np.float16).tobytes()).decode()
    w['b%d'%i] = base64.b64encode(b.astype(np.float16).tobytes()).decode()
    w['s%d'%i] = list(W.shape)
json.dump(w, open(sys.argv[2],'w'))
# reference vectors for checking the JS forward pass
ref = {'x': [round(float(v),4) for v in Xte[0].reshape(-1)], 'p': [float(v) for v in m.predict_proba(Xte[:1].reshape(1,-1))[0]], 'y': int(yte[0])}
json.dump(ref, open(sys.argv[3],'w'))
