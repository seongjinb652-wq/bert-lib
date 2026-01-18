"""
File: model_card.py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Hub에 업로드할 모델 카드(README.md)를 생성하고
    로컬에 저장하는 스크립트입니다. 모델의 설명, 학습 데이터, 학습 절차,
    평가 결과, 사용법, 편향 및 윤리적 고려사항을 포함한 템플릿을 출력하고
    파일로 저장할 수 있습니다.

Features:
    - 모델 카드 템플릿 정의 (언어, 라이선스, 데이터셋, 태그 등)
    - 모델 설명, Intended Uses, How to Use, Training Data, Evaluation Results 포함
    - 로컬에 README.md 파일로 저장하는 방법 예시 제공

Dependencies:
    - Python 3.x (표준 라이브러리만 사용)

Usage:
    $ python 99_model_card.py
    → 모델 카드 템플릿 출력 및 README.md 파일 저장 방법 확인 가능

Note:
    - 저장된 README.md 파일은 Hugging Face Hub에 push_to_hub() 또는 git push로 업로드 가능
    - 모델 카드 내용은 프로젝트 목적과 데이터셋에 맞게 수정 가능
"""
print("=" * 60)
print("📝 모델 카드 템플릿")
print("=" * 60)

model_card_template = """
---
language: ko
license: mit
library_name: transformers
datasets:
- nsmc
tags:
- text-classification
- sentiment-analysis
pipeline_tag: text-classification
---

# NSMC 감성 분석 모델

## Model Description

이 모델은 `klue/bert-base`를 기반으로 NSMC(Naver Sentiment Movie Corpus) 데이터셋으로
파인튜닝한 한국어 감성 분석 모델입니다.

## Intended Uses & Limitations

### 적합한 용도
- 한국어 영화 리뷰의 긍정/부정 분류
- 한국어 짧은 텍스트의 감성 분석

### 제한사항
- 영화 리뷰 이외의 도메인에서는 성능이 저하될 수 있습니다
- 긴 문서보다 짧은 문장에 최적화되어 있습니다

## How to Use

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="YOUR_USERNAME/nsmc-finetuned-bert")
result = classifier("이 영화 정말 재미있어요!")
print(result)  # [{'label': 'positive', 'score': 0.99}]
```

## Training Data

- 데이터셋: NSMC (Naver Sentiment Movie Corpus)
- 훈련 샘플: 150,000개
- 테스트 샘플: 50,000개

## Training Procedure

- Base model: klue/bert-base
- Learning rate: 2e-5
- Batch size: 32
- Epochs: 3

## Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.89 |
| F1 Score | 0.89 |

## Bias & Ethical Considerations

- 이 모델은 영화 리뷰 데이터로만 학습되어 다른 도메인의 텍스트에 편향된 결과를 보일 수 있습니다
- 비속어나 혐오 표현이 포함된 리뷰로 학습되었을 수 있으므로 주의가 필요합니다
"""

print(model_card_template)

# 모델 카드를 파일로 저장하는 예시
print("\n" + "=" * 60)
print("💾 모델 카드 저장 방법")
print("=" * 60)

print("""
# 로컬에 저장
with open("./my-model/README.md", "w") as f:
    f.write(model_card_template)

# 이후 push_to_hub() 또는 git push로 업로드
""")
