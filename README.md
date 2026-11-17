# 예나의 연습장 (웹)

아이패드 Safari에서 쓰는 수학 연습 웹 앱. 파일 하나(`index.html`)로 동작한다.

| 파일 | 내용 |
|---|---|
| `index.html` | 완성본. 이 파일만 올리면 된다 |
| `template.html` | 화면과 동작 코드. 고칠 때는 이 파일을 고친다 |
| `rec.js` | 필기 숫자 인식 (획 나누기, 28x28 변환, 모델 계산) |
| `weights.json` | 숫자 인식 모델 가중치 |
| `build.py` | `template.html` + `rec.js` + `weights.json` → `index.html` |
| `train.py` | 모델 학습 스크립트 (MNIST 필요) |
| `handwriting-test.html` | 필기감·인식률 테스트 페이지 |

고친 뒤에는 `python3 build.py` 로 `index.html` 을 다시 만든다.

기록(오늘 진행, 스티커, 다시 풀기)은 브라우저 localStorage 에 저장된다.

## 단계 추가

`template.html` 의 `<script type="application/json" id="levelsData">` 안에 한 덩어리를 붙이고 `python3 build.py` 를 실행한다.

| 항목 | 뜻 |
|---|---|
| `id` | 단계 구분용 이름 (겹치면 안 됨) |
| `title`, `desc` | 화면에 보이는 이름과 설명 |
| `op` | `+` 또는 `-` |
| `digits` | 앞 수와 뒤 수의 자릿수. 예: `[2, 1]` |
| `shuffle` | `true` 면 앞뒤 순서를 섞는다 |
| `max` | 더하기는 합의 상한, 빼기는 큰 수의 상한 |
| `carry` | 받아올림·받아내림: `any`, `none`(없는 것만), `only`(있는 것만) |
| `problems` | 규칙 대신 문제를 직접 적을 때. 예: `[[37, 5], [8, 46], [60, 4, "-"]]` |

오늘 할 단계와 이름은 홈의 "부모님 설정"에서 바꾼다.
