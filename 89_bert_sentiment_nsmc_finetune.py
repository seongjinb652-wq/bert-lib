"""
File: bert_sentiment_nsmc_finetune.py .py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Transformers의 Trainer를 활용하여
    NSMC(Naver Sentiment Movie Corpus) 데이터셋으로
    감성 분석 모델을 파인튜닝하고 성능을 평가하는 스크립트입니다.

Features:
    - TrainingArguments 설정 (출력 디렉토리, 학습률, 배치 크기, 에폭 등)
    - compute_metrics 함수 정의 (Accuracy, F1 점수)
    - AutoModelForSequenceClassification 로드 (긍정/부정 2개 레이블)
    - Trainer 객체 생성 및 학습/검증 데이터셋 연결
    - 학습 전 베이스라인 성능 평가
    - 파인튜닝 학습 실행 및 학습 후 성능 평가
    - 정확도 향상 계산 및 출력

Dependencies:
    - transformers
    - datasets
    - evaluate
    - numpy
    - torch

Usage:
    $ python bert_sentiment_nsmc_finetune.py 
    → 학습 전후 성능 비교 결과를 확인 가능

Note:
    - checkpoint 변수에 사전학습 모델 이름을 지정해야 함 (예: "klue/bert-base")
    - tokenized_datasets는 사전에 토큰화된 NSMC 데이터셋이어야 함
"""
from transformers import TrainingArguments

training_args = TrainingArguments(
    # 📁 출력 디렉토리
    output_dir="./nsmc-finetuned-bert",
    # 📊 학습 설정
    num_train_epochs=1,  # 전체 데이터를 1번 학습 (시간 절약)
    per_device_train_batch_size=32,  # 배치 크기 (GPU 메모리에 따라 조절)
    per_device_eval_batch_size=64,  # 평가 배치 크기
    # 📈 평가 설정
    eval_strategy="epoch",  # 매 에폭마다 평가
    save_strategy="epoch",  # 매 에폭마다 저장
    # ⚙️ 최적화 설정
    learning_rate=2e-5,  # 학습률 (BERT 권장값)
    weight_decay=0.01,  # 가중치 감쇠
    # 📝 로깅
    logging_steps=500,  # 500스텝마다 로그 출력
    # 🔧 기타
    load_best_model_at_end=True,  # 학습 후 가장 좋은 모델 로드
    metric_for_best_model="accuracy",  # 최고 모델 기준
    # ⚡ 성능 최적화 (GPU 사용 시)
    # fp16=True,                     # 혼합 정밀도 학습
)

print("✅ TrainingArguments 설정 완료!")
print(f"   출력 디렉토리: {training_args.output_dir}")
print(f"   학습 에폭: {training_args.num_train_epochs}")
print(f"   배치 크기: {training_args.per_device_train_batch_size}")
print(f"   학습률: {training_args.learning_rate}")      import numpy as np
import evaluate

# 평가지표 로드
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """
    평가지표 계산 함수

    Args:
        eval_pred: (logits, labels) 튜플

    Returns:
        dict: {"accuracy": ..., "f1": ...}
    """
    logits, labels = eval_pred

    # logits에서 예측 클래스 추출
    predictions = np.argmax(logits, axis=-1)

    # 정확도 계산
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)

    # F1 점수 계산
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")

    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


print("✅ 평가지표 함수 정의 완료!")   from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

# 모델 로드 (2개 레이블: 긍정/부정)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2,
    id2label={0: "부정", 1: "긍정"},
    label2id={"부정": 0, "긍정": 1},
)

print("📦 모델 로드 완료!")
print(f"   레이블 매핑: {model.config.id2label}")     # 데이터 콜레이터 (동적 패딩)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer 설정 완료!")     # 학습 전 베이스라인 평가
print("📊 학습 전 모델 성능 (베이스라인):")
baseline_results = trainer.evaluate()
print(f"   Accuracy: {baseline_results['eval_accuracy']:.4f}")
print(f"   F1 Score: {baseline_results['eval_f1']:.4f}")    # 🚀 학습 시작!
print("\n🚀 파인튜닝 시작!")
print("=" * 50)

train_result = trainer.train()

print("=" * 50)
print("✅ 파인튜닝 완료!")     # 학습 후 성능 평가
print("\n📊 학습 후 모델 성능:")
final_results = trainer.evaluate()
print(f"   Accuracy: {final_results['eval_accuracy']:.4f}")
print(f"   F1 Score: {final_results['eval_f1']:.4f}")

# 성능 향상 계산
acc_improvement = final_results["eval_accuracy"] - baseline_results["eval_accuracy"]
print(f"\n📈 정확도 향상: {acc_improvement:+.4f}")  
