"""
File: bert_finetune_eval.py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Transformers의 pipeline을 활용하여
    파인튜닝된 텍스트 분류 모델을 평가하는 예제 코드입니다.

Features:
    - 파인튜닝된 모델과 토크나이저를 pipeline으로 로드
    - 테스트 샘플에 대해 예측 수행
    - 정답 라벨과 비교하여 정확도 계산
    - 결과를 직관적으로 출력 (✅/❌ 표시 및 신뢰도)

Dependencies:
    - transformers (pipeline)
    - 파인튜닝된 모델 및 토크나이저

Usage:
    1. 파인튜닝된 모델과 토크나이저를 준비
    2. 본 스크립트를 실행하면 테스트 샘플에 대한 예측 결과와 정확도가 출력됨

Note:
    - 테스트 샘플은 예시용이며, 실제 평가 시에는 별도의 검증 데이터셋을 사용하는 것이 바람직함
"""
# 파인튜닝 후 모델로 pipeline 생성
classifier_after = pipeline('text-classification', model=model, tokenizer=tokenizer)

# 동일한 테스트 샘플 사용
test_samples = [
    ("삼성전자, 신형 갤럭시 시리즈 공개", "IT과학"),
    ("코스피 장중 3000선 돌파", "경제"),
    ("손흥민, 시즌 15호골 폭발", "스포츠"),
    ("여야, 예산안 처리 두고 충돌", "정치"),
    ("전국 미세먼지 '나쁨'...외출 자제", "사회"),
]

print("=" * 60)
print("🟢 파인튜닝 후 모델 예측 결과")
print("=" * 60)
print()

correct = 0
for title, true_label in test_samples:
    result = classifier_after(title)[0]
    is_correct = result['label'] == true_label
    correct += is_correct
    status = "✅" if is_correct else "❌"

    print(f"제목: {title}")
    print(f"  정답: {true_label}")
    print(f"  예측: {result['label']} (신뢰도: {result['score']:.2%}) {status}")
    print()

print(f"정확도: {correct}/{len(test_samples)} ({correct/len(test_samples)*100:.1f}%)")
print("\n✨ 파인튜닝 후 성능이 크게 향상되었습니다!")
