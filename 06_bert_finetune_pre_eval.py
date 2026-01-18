"""
File: bert_finetune_pre_eval.py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Transformers의 AutoModelForSequenceClassification을 활용하여
    파인튜닝 전 텍스트 분류 모델의 성능을 평가하는 예제 코드입니다.

Features:
    - raw 데이터셋에서 라벨 이름 추출
    - id2label, label2id 매핑 생성
    - 파인튜닝 전 모델을 pipeline으로 로드
    - 테스트 샘플에 대해 예측 수행 및 정확도 계산
    - 랜덤 초기화된 분류 헤드로 인해 낮은 성능 확인

Dependencies:
    - transformers (AutoModelForSequenceClassification, pipeline)
    - datasets (raw 데이터셋)
    - numpy (argmax 등 필요 시)

Usage:
    1. 체크포인트와 토크나이저를 준비
    2. 본 스크립트를 실행하면 파인튜닝 전 모델의 예측 결과와 정확도가 출력됨
    3. 성능은 랜덤 수준에 가까우며, 파인튜닝 필요성을 확인 가능

Note:
    - 출력된 정확도는 베이스라인 성능으로, 이후 파인튜닝 후 성능과 비교하는 데 활용
"""

#  raw['train'].features['label'].names에서 라벨 이름 추출 가능
# label_names = _______________
label_names = raw['train'].features['label'].names

# 숫자 → 라벨 이름
id2label = {i: n for i, n in enumerate(label_names)}

# 라벨 이름 → 숫자
label2id = {n: i for i, n in enumerate(label_names)}

print("라벨 목록:", label_names)
print("id2label:", id2label

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint ,         # 체크포인트
    num_labels=len(label_names),  # 분류할 클래스 수
    id2label=id2label,    # 숫자 → 라벨
    label2id=label2id     # 라벨 → 숫자
)

from transformers import pipeline

# 파인튜닝 전 모델로 pipeline 생성
classifier_before = pipeline('text-classification', model=model, tokenizer=tokenizer)

# 테스트할 뉴스 제목들 (정답 포함)
test_samples = [
    ("삼성전자, 신형 갤럭시 시리즈 공개", "IT과학"),
    ("코스피 장중 3000선 돌파", "경제"),
    ("손흥민, 시즌 15호골 폭발", "스포츠"),
    ("여야, 예산안 처리 두고 충돌", "정치"),
    ("전국 미세먼지 '나쁨'...외출 자제", "사회"),
]

print("=" * 60)
print("🔴 파인튜닝 전 모델 예측 결과")
print("=" * 60)
print("\n⚠️  분류 헤드가 랜덤 초기화 상태이므로 예측이 부정확합니다!\n")

correct = 0
for title, true_label in test_samples:
    result = classifier_before(title)[0]
    is_correct = result['label'] == true_label
    correct += is_correct
    status = "✅" if is_correct else "❌"

    print(f"제목: {title}")
    print(f"  정답: {true_label}")
    print(f"  예측: {result['label']} (신뢰도: {result['score']:.2%}) {status}")
    print()

print(f"정확도: {correct}/{len(test_samples)} ({correct/len(test_samples)*100:.1f}%)")
print("\n→ 랜덤 수준(약 14.3% = 1/7)에 가까운 성능!")
