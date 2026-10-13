# SKALA AutoInsight

KOSPI 지수와 건강식품 지수의 D+3 상승 가능성을 예측하는 다중 타깃 분류 프로젝트입니다.

## 프로젝트 구조

```text
.
├── data/
│   ├── raw/            # 원본 학습·평가 데이터
│   └── submissions/    # 제출 형식 예시
├── docs/               # 변수 설명서 등 참고 문서
├── notebooks/          # 탐색 및 모델링 노트북
├── outputs/
│   ├── figures/        # 생성된 시각화
│   ├── models/         # 저장된 모델
│   └── submissions/    # 생성된 제출 파일
└── src/                # 재사용 가능한 분석 코드
```

## 실행 방법

Python 가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

`notebooks/01_baseline.ipynb`를 프로젝트 루트 또는 `notebooks/` 디렉터리를 작업 경로로 실행할 수 있습니다. 생성된 제출 파일은 `outputs/submissions/`에 저장됩니다.

## 데이터 관리 원칙

- `data/raw/`의 원본 파일은 직접 수정하지 않습니다.
- 모델, 그림, 제출 파일 등 재생성 가능한 결과물은 `outputs/`에 저장합니다.
- 반복 사용하는 전처리 및 학습 코드는 `src/`로 분리합니다.
- Public/Private 평가 데이터는 학습 및 전처리 통계 산출에 사용하지 않습니다.
