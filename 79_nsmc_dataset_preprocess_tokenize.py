"""
File: nsmc_dataset_preprocess_tokenize.py
Author: 성진
Date: 2026-01-18

Description:
    NSMC(Naver Sentiment Movie Corpus) 데이터셋을 로드하고,
    전처리 및 토큰화를 수행하여 Hugging Face Trainer에 입력할 수 있는
    최종 PyTorch 텐서 형식으로 변환하는 스크립트입니다.

Features:
    - NSMC 데이터셋 로드 (train/test CSV 파일)
    - 데이터 통계 및 레이블 분포 확인
    - 결측치(None) 제거
    - AutoTokenizer를 이용한 문장 토큰화 (padding, truncation, max_length=128)
    - 토큰화 결과 확인 (input_ids, labels)
    - 불필요한 컬럼 제거 및 레이블 컬럼 이름 변경
    - PyTorch 텐서 형식으로 변환

Dependencies:
    - datasets
    - transformers
    - torch
    - collections (Counter)

Usage:
    $ python nsmc_dataset_preprocess_tokenize.py
    → 데이터셋 로드, 전처리, 토큰화 결과를 확인 가능

Note:
    - checkpoint 변수에 사용할 사전학습 모델 지정 필요 (예: "klue/bert-base")
    - 최종 결과(tokenized_datasets)는 Trainer 학습 단계에서 사용됨
"""
from datasets import load_dataset

# NSMC 원본 데이터(GitHub) 주소 설정
data_files = {
    "train": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
    "test": "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
}

# CSV 로더를 사용하여 로드 (구분자는 탭 '\t')
raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t")

print("📦 데이터셋 구조:")
print(raw_datasets)

print("\n📊 학습 데이터 샘플:")
# 샘플 출력 (컬럼명: id, document, label)
print(raw_datasets["train"][0])
print(raw_datasets["train"][1])  # 데이터 통계 확인
print("📈 데이터셋 통계:")
print(f"   학습 데이터: {len(raw_datasets['train']):,}개")
print(f"   테스트 데이터: {len(raw_datasets['test']):,}개")

# 레이블 분포 확인
from collections import Counter

train_labels = raw_datasets["train"]["label"]
label_counts = Counter(train_labels)
print(f"\n🏷️ 레이블 분포:")
print(f"   부정(0): {label_counts[0]:,}개 ({label_counts[0] / len(train_labels):.1%})")
print(f"   긍정(1): {label_counts[1]:,}개 ({label_counts[1] / len(train_labels):.1%})")  # 1. 결측치(None)가 있는 행 제거
print(f"전처리 전 데이터 개수: {len(raw_datasets['train'])}")

# document 컬럼이 None이 아닌 것만 남김
raw_datasets = raw_datasets.filter(lambda x: x["document"] is not None)

print(f"전처리 후 데이터 개수: {len(raw_datasets['train'])}")  from transformers import AutoTokenizer

checkpoint = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# 토큰화 함수 정의
def tokenize_function(examples):
    """
    데이터셋의 'document' 컬럼을 토큰화
    """
    return tokenizer(
        examples["document"],
        padding="max_length",  # 최대 길이로 패딩
        truncation=True,  # 길면 자르기
        max_length=128,  # 최대 128 토큰
    )


# 전체 데이터셋에 토큰화 적용
# batched=True: 여러 샘플을 한 번에 처리 (빠름!)
print("🔄 토큰화 진행 중...")
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, desc="Tokenizing"
)

print("\n✅ 토큰화 완료!")
print(f"   컬럼: {tokenized_datasets['train'].column_names}")    # 토큰화 결과 확인
sample = tokenized_datasets["train"][0]
print("📝 토큰화 결과 예시:")
print(f"   원본: {raw_datasets['train'][0]['document'][:50]}...")
print(f"   input_ids 길이: {len(sample['input_ids'])}")
print(f"   레이블: {sample['label']}")  # 불필요한 컬럼 제거
tokenized_datasets = tokenized_datasets.remove_columns(["id", "document"])

# 레이블 컬럼 이름 변경 (Trainer가 기대하는 형식)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# PyTorch 텐서 형식으로 설정
tokenized_datasets.set_format("torch")

print("📦 최종 데이터셋 형태:")
print(f"   컬럼: {tokenized_datasets['train'].column_names}")
print(f"   타입: {type(tokenized_datasets['train'][0]['input_ids'])}")   
