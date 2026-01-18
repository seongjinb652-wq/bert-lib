"""
File: bert_finetune_train.py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Transformers의 Trainer를 활용하여
    파인튜닝된 텍스트 분류 모델을 학습하는 예제 코드입니다.

Features:
    - Trainer 객체 생성 (모델, 학습 설정, 데이터셋, 토크나이저, 메트릭 포함)
    - train_dataset과 eval_dataset을 사용하여 학습 및 검증 수행
    - compute_metrics 함수를 통해 정확도, F1 등 성능 지표 계산

Dependencies:
    - transformers (Trainer, TrainingArguments 등)
    - 토큰화된 데이터셋 (train, validation)
    - 사용자 정의 compute_metrics 함수

Usage:
    1. 학습 데이터셋과 검증 데이터셋을 준비 및 토큰화
    2. TrainingArguments를 설정 (출력 디렉토리, 에폭, 배치 크기 등)
    3. 본 스크립트를 실행하면 모델 학습이 시작되고 결과가 저장됨

Note:
    - 학습 결과는 output_dir에 저장되며, 이후 평가 및 추론에 활용 가능
"""

trainer = Trainer(
    model=model,           # 모델
    args=training_args,            # 학습 설정
    train_dataset=tokenized['train'],   # 학습 데이터 (토큰화된)
    eval_dataset=tokenized['validation'],    # 검증 데이터 (토큰화된)
    data_collator=data_collator,   # 패딩 처리
    tokenizer=tokenizer,       # 토크나이저
    compute_metrics=compute_metrics  # 메트릭 함수
)
trainer.train()
