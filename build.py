# template.html + rec.js + weights.json -> index.html (파일 하나로 합치기)
# 사용: python3 build.py
t = open('template.html', encoding='utf-8').read()
rec = open('rec.js', encoding='utf-8').read().replace(
    "if (typeof module !== 'undefined') module.exports",
    "if (typeof module !== 'undefined' && module.exports) module.exports")
w = open('weights.json', encoding='utf-8').read()
open('index.html', 'w', encoding='utf-8').write(t.replace('/*REC*/', rec).replace('/*WEIGHTS*/', w))
print('index.html written')
