"""
File: sentiment_and_topic_model_save_and_ynat.py
Author: 성진
Date: 2026-01-18

Description:
    NSMC 감성 분석 모델을 로컬에 저장하고, 저장된 모델을 불러와
    영화 리뷰 감성 분석을 테스트한 뒤 KLUE YNAT 데이터셋을 로드하여
    뉴스 토픽 분류 파인튜닝을 준비하는 스크립트입니다.

Features:
    - trainer.save_model()과 tokenizer.save_pretrained()로 모델 저장
    - 저장된 파일 목록과 크기 확인
    - pipeline을 이용한 감성 분석 테스트 (긍정/부정)
    - KLUE YNAT 데이터셋 로드 및 샘플 출력
    - YNAT 레이블 종류 확인 (7개 토픽)
    - YNAT 파인튜닝 코드 예시 제공 (주석 처리)

Dependencies:
    - transformers
    - datasets
    - torch
    - os (파일 확인용)

Usage:
    $ python 97_sentiment_and_topic_model_save_and_ynat.py
    → 모델 저장, 감성 분석 테스트, YNAT 데이터셋 준비 과정을 실행

Note:
    - YNAT 파인튜닝 코드는 주석 처리되어 있으며, 실제 학습 시 주석을 해제해야 함
    - 저장된 모델은 이후 Hub 업로드나 추가 파인튜닝에 활용 가능
"""
# 모델 저장
save_path = "./my-nsmc-model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 모델이 '{save_path}'에 저장되었습니다!")

# 저장된 파일 확인
import os

files = os.listdir(save_path)
print(f"\n📁 저장된 파일들:")
for f in files:
    size = os.path.getsize(os.path.join(save_path, f))
    print(
        f"   {f}: {size / 1024 / 1024:.1f} MB"
        if size > 1024 * 1024
        else f"   {f}: {size / 1024:.1f} KB"
    )    from transformers import pipeline

# 저장된 모델로 pipeline 생성
my_classifier = pipeline("sentiment-analysis", model=save_path, tokenizer=save_path)

# 테스트!
test_reviews = [
    "이 영화 진짜 최고예요! 감동받았습니다.",
    "시간 낭비했네요. 별로입니다.",
    "그냥 그래요. 평범한 영화입니다.",
    "배우 연기가 정말 인상적이었어요!",
    "스토리가 너무 지루했어요.",
]

print("🎬 영화 리뷰 감성 분석 결과:")
print("-" * 50)
for review in test_reviews:
    result = my_classifier(review)[0]
    emoji = "😊" if result["label"] == "긍정" else "😠"
    print(f"{emoji} {review[:25]}...")
    print(f"   → {result['label']} ({result['score']:.2%})")
    print()     # KLUE YNAT 데이터셋 로드
ynat_datasets = load_dataset("klue", "ynat")

print("📰 YNAT 데이터셋 (뉴스 토픽 분류):")
print(f"   학습 데이터: {len(ynat_datasets['train']):,}개")
print(f"   검증 데이터: {len(ynat_datasets['validation']):,}개")

# 샘플 확인
sample = ynat_datasets["train"][0]
print(f"\n📝 샘플:")
print(f"   제목: {sample['title']}")
print(f"   레이블: {sample['label']}")

# 레이블 종류 확인
label_names = ["IT/과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]
print(f"\n🏷️ 7개 토픽: {label_names}"  # YNAT 파인튜닝 코드 (실행은 주석 처리)
"""
# 토큰화 함수
def tokenize_ynat(examples):
    return tokenizer(
        examples["title"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

# 전처리
tokenized_ynat = ynat_datasets.map(tokenize_ynat, batched=True)
tokenized_ynat = tokenized_ynat.remove_columns(["guid", "title", "url", "date"])
tokenized_ynat = tokenized_ynat.rename_column("label", "labels")
tokenized_ynat.set_format("torch")

# 모델 (7개 클래스)
ynat_model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=7
)

# Trainer
ynat_trainer = Trainer(
    model=ynat_model,
    args=TrainingArguments(
        output_dir="./ynat-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        evaluation_strategy="epoch"
    ),
    train_dataset=tokenized_ynat["train"],
    eval_dataset=tokenized_ynat["validation"],
    compute_metrics=compute_metrics
)

# 학습
ynat_trainer.train()
"""

print("💡 YNAT 파인튜닝 코드가 준비되었습니다.")
print("   실제 실행하려면 주석을 해제하세요!") 
