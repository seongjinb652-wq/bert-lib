"""
File: bert_finetune_metrics.py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Trainer를 활용하여 파인튜닝된 텍스트 분류 모델의
    최종 성능을 평가하고 Accuracy 및 F1 점수를 출력하는 예제 코드입니다.

Features:
    - trainer.evaluate()를 통해 평가 지표 계산
    - Accuracy와 Macro F1 점수를 포맷팅하여 출력
    - 결과를 직관적으로 확인할 수 있도록 콘솔에 표시

Dependencies:
    - transformers (Trainer)
    - 파인튜닝된 모델 및 데이터셋

Usage:
    1. 파인튜닝된 모델과 Trainer 객체를 준비
    2. 본 스크립트를 실행하면 최종 평가 결과가 출력됨

Note:
    - 출력되는 지표는 검증 데이터셋 기준이며,
      실제 테스트셋 평가 시에는 별도의 데이터셋을 지정해야 함
"""
# metrics = _______________
metrics = trainer.evaluate()

print("\n=== 최종 평가 결과 ===")
print(f"정확도 (Accuracy): {metrics['eval_accuracy']:.4f}")
print(f"F1 점수 (Macro):   {metrics['eval_f1']:.4f}")
