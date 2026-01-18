"""
File: bert_finetune_config.py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Transformers의 Trainer를 활용한
    텍스트 분류 모델 파인튜닝을 위한 학습 설정 스크립트입니다.

Features:
    - DataCollatorWithPadding을 사용하여 배치 내 패딩 처리
    - Accuracy 및 Macro F1 메트릭 계산 함수 정의
    - TrainingArguments 설정 (배치 크기, 학습률, 에폭, 로깅 등)
    - 최고 성능 모델을 F1 기준으로 저장 및 로드

Dependencies:
    - transformers (Trainer, TrainingArguments, DataCollatorWithPadding)
    - evaluate (accuracy, f1)
    - numpy (argmax)

Usage:
    1. 토크나이저와 데이터셋을 준비
    2. 본 스크립트에서 정의된 data_collator, compute_metrics, training_args를 Trainer에 전달
    3. trainer.train() 실행으로 학습 시작

Note:
    - output_dir에 학습 결과 및 체크포인트가 저장됨
    - compute_metrics 함수는 검증 데이터셋 기준으로 성능을 평가
"""

# data_collator = _______________
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 메트릭 로드
acc_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')

# TODO: 메트릭 계산 함수 정의
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # TODO: logits에서 예측값 추출 (argmax 사용)
    # preds = _______________
    preds =  np.argmax(logits, axis=-1)


    # 정확도 계산
    acc = acc_metric.compute(predictions=preds, references=labels)['accuracy']

    # F1 점수 계산 (macro average)
    f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')['f1']

    return {'accuracy': acc, 'f1': f1}

training_args = TrainingArguments(
    # output_dir=_______________,              # 결과 저장 디렉토리
    output_dir='klue-ynat-finetuned',

    # 배치 및 학습 설정
    per_device_train_batch_size=32,         # 학습 배치 크기
    per_device_eval_batch_size=64,          # 평가 배치 크기
    num_train_epochs=3,                    # 총 에폭 수
    learning_rate=3e-5,                       # 학습률 (예: 3e-5)

    # 정규화
    weight_decay=0.01,
    warmup_ratio=0.1,

    # 평가 및 저장 전략
    eval_strategy='epoch',           # 'epoch' 또는 'steps'
    save_strategy='epoch',           # 'epoch' 또는 'steps'
    load_best_model_at_end=True,            # 최고 성능 모델 로드
    metric_for_best_model='f1',              # 최고 모델 선정 기준

    # 로깅
    logging_steps=50,
    report_to='none',

    # 재현성
    seed=42
)
