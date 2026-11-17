// ---- 숫자 인식 (DOM 없이 동작) ----
function f16ToF32(b64) {
  var bin = atob(b64), n = bin.length >> 1, out = new Float32Array(n);
  for (var i = 0; i < n; i++) {
    var h = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    var s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (f / 1024)
           : e === 31 ? (f ? NaN : s * Infinity)
           : s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}
function loadModel(w) {
  var layers = [];
  for (var i = 0; w['W' + i]; i++) {
    layers.push({ W: f16ToF32(w['W' + i]), b: f16ToF32(w['b' + i]), nin: w['s' + i][0], nout: w['s' + i][1] });
  }
  return layers;
}
function predict(layers, x) {
  var a = x;
  for (var l = 0; l < layers.length; l++) {
    var L = layers[l], z = new Float32Array(L.nout);
    for (var j = 0; j < L.nout; j++) z[j] = L.b[j];
    for (var i = 0; i < L.nin; i++) {
      var v = a[i];
      if (v === 0) continue;
      var row = i * L.nout;
      for (var k = 0; k < L.nout; k++) z[k] += v * L.W[row + k];
    }
    if (l < layers.length - 1) for (var q = 0; q < L.nout; q++) if (z[q] < 0) z[q] = 0;
    a = z;
  }
  var m = -Infinity, sum = 0, p = new Float32Array(a.length);
  for (var c = 0; c < a.length; c++) if (a[c] > m) m = a[c];
  for (c = 0; c < a.length; c++) { p[c] = Math.exp(a[c] - m); sum += p[c]; }
  for (c = 0; c < a.length; c++) p[c] /= sum;
  return p;
}
function bbox(stroke) {
  var x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (var i = 0; i < stroke.length; i++) {
    var p = stroke[i];
    if (p.x < x0) x0 = p.x; if (p.x > x1) x1 = p.x;
    if (p.y < y0) y0 = p.y; if (p.y > y1) y1 = p.y;
  }
  return { x0: x0, y0: y0, x1: x1, y1: y1 };
}
// 획을 가로 위치로 묶어 숫자 하나씩으로 나눈다.
function groupStrokes(strokes) {
  var items = strokes.filter(function (s) { return s.length > 0; })
    .map(function (s) { return { s: [s], b: bbox(s) }; })
    .sort(function (a, b) { return a.b.x0 - b.b.x0; });
  var groups = [];
  items.forEach(function (it) {
    var g = groups[groups.length - 1];
    if (g) {
      // 세로 한 줄처럼 폭이 없는 획도 겹침을 잴 수 있게 양쪽으로 3px 넓혀서 본다
      var overlap = Math.min(g.b.x1, it.b.x1) - Math.max(g.b.x0, it.b.x0) + 6;
      var minW = Math.min(g.b.x1 - g.b.x0, it.b.x1 - it.b.x0) + 6;
      if (overlap > 0.3 * minW) {
        g.s = g.s.concat(it.s);
        g.b = { x0: Math.min(g.b.x0, it.b.x0), y0: Math.min(g.b.y0, it.b.y0), x1: Math.max(g.b.x1, it.b.x1), y1: Math.max(g.b.y1, it.b.y1) };
        return;
      }
    }
    groups.push({ s: it.s, b: it.b });
  });
  return groups;
}
// 한 묶음을 MNIST 형식(28x28, 20px 상자, 무게중심 가운데)으로 그린다.
function rasterize(group) {
  var S = 112, R = 5.2, buf = new Float32Array(S * S);
  var w = group.b.x1 - group.b.x0, h = group.b.y1 - group.b.y0;
  var size = Math.max(w, h, 1), k = (80 - 2 * R) / size;
  var ox = (S - w * k) / 2 - group.b.x0 * k, oy = (S - h * k) / 2 - group.b.y0 * k;
  function seg(ax, ay, bx, by) {
    var minx = Math.max(0, Math.floor(Math.min(ax, bx) - R - 1)), maxx = Math.min(S - 1, Math.ceil(Math.max(ax, bx) + R + 1));
    var miny = Math.max(0, Math.floor(Math.min(ay, by) - R - 1)), maxy = Math.min(S - 1, Math.ceil(Math.max(ay, by) + R + 1));
    var dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    for (var y = miny; y <= maxy; y++) for (var x = minx; x <= maxx; x++) {
      var t = len2 ? ((x - ax) * dx + (y - ay) * dy) / len2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      var ex = x - (ax + t * dx), ey = y - (ay + t * dy);
      var v = R + 0.5 - Math.sqrt(ex * ex + ey * ey);
      if (v > 1) v = 1;
      if (v > buf[y * S + x]) buf[y * S + x] = v;
    }
  }
  group.s.forEach(function (st) {
    if (st.length === 1) seg(st[0].x * k + ox, st[0].y * k + oy, st[0].x * k + ox, st[0].y * k + oy);
    for (var i = 1; i < st.length; i++) seg(st[i - 1].x * k + ox, st[i - 1].y * k + oy, st[i].x * k + ox, st[i].y * k + oy);
  });
  var small = new Float32Array(784), mx = 0, my = 0, mass = 0;
  for (var y = 0; y < 28; y++) for (var x = 0; x < 28; x++) {
    var a = 0;
    for (var j = 0; j < 4; j++) for (var i = 0; i < 4; i++) a += buf[(y * 4 + j) * S + x * 4 + i];
    a /= 16; small[y * 28 + x] = a; mass += a; mx += a * x; my += a * y;
  }
  if (!mass) return small;
  var sx = Math.round(13.5 - mx / mass), sy = Math.round(13.5 - my / mass), out = new Float32Array(784);
  for (y = 0; y < 28; y++) for (x = 0; x < 28; x++) {
    var px = x - sx, py = y - sy;
    if (px >= 0 && px < 28 && py >= 0 && py < 28) out[y * 28 + x] = small[py * 28 + px];
  }
  return out;
}
// 획 전체를 읽어 { text, digits:[{d, p, img}] } 로 돌려준다.
function readNumber(layers, strokes) {
  var groups = groupStrokes(strokes).filter(function (g) {
    return Math.max(g.b.x1 - g.b.x0, g.b.y1 - g.b.y0) >= 10;   // 실수로 찍힌 점은 버린다
  });
  var digits = groups.map(function (g) {
    var img = rasterize(g), p = predict(layers, img), best = 0;
    for (var i = 1; i < 10; i++) if (p[i] > p[best]) best = i;
    return { d: best, p: p[best], img: img };
  });
  return { text: digits.map(function (x) { return x.d; }).join(''), digits: digits };
}
if (typeof module !== 'undefined') module.exports = { loadModel: loadModel, predict: predict, groupStrokes: groupStrokes, rasterize: rasterize, readNumber: readNumber };
