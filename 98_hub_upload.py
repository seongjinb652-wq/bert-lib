"""
File: hub_upload.py
Author: 성진
Date: 2026-01-18

Description:
    Hugging Face Transformers와 huggingface_hub를 활용하여
    한국어 BERT/RoBERTa 기반 모델을 로드하고 [MASK] 토큰 예측을 수행한 뒤,
    Hugging Face Hub에 업로드하는 예제 코드입니다.

Features:
    - klue/roberta-small 모델과 토크나이저 로드
    - [MASK] 토큰 예측 결과 확인 (top-k 후보 출력)
    - 모델 설정 정보 출력 (히든 크기, 어텐션 헤드 수, 레이어 수, 어휘 크기)
    - Hugging Face Hub 로그인 안내
    - Trainer 기반 자동 업로드 설정 (push_to_hub=True)
    - push_to_hub() 메소드로 수동 업로드 예시 제공

Dependencies:
    - transformers
    - torch
    - huggingface_hub

Usage:
    1. Hugging Face Hub 토큰 발급 및 notebook_login() 실행
    2. Trainer로 학습 시 push_to_hub=True 옵션을 통해 자동 업로드
    3. 또는 model.push_to_hub(), tokenizer.push_to_hub()로 수동 업로드 가능

Note:
    - 'YOUR_HF_USERNAME'을 자신의 Hugging Face 사용자명으로 변경해야 함
    - repo_id는 "사용자명/모델이름" 형식으로 지정
"""

# 필요한 라이브러리 설치
# !pip install transformers torch -q  import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 1. 모델과 토크나이저 로드
checkpoint = "klue/roberta-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(checkpoint)

# 2. 입력 문장 준비
test_sentence = "대한민국의 수도는 [MASK]이다."
print(f"📝 입력 문장: {test_sentence}")

# 3. 토큰화 (PyTorch 텐서로 변환)
inputs = tokenizer(test_sentence, return_tensors="pt")

# 4. [MASK] 토큰의 위치(인덱스) 찾기
# 입력된 문장 내에서 mask_token_id를 가진 위치를 찾습니다.
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

# 5. 모델 추론 (Forward Pass)
# 모델에 입력을 넣고 예측값을 받습니다.
with torch.no_grad():
    outputs = model(**inputs)

    # 여기서 튜플로 나오든 딕셔너리로 나오든 상관없이 첫 번째 요소(Logits)를 가져옵니다.
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs.logits

# 6. 결과 확인
# [MASK] 위치의 로짓(점수)을 가져옵니다.
mask_token_logits = logits[0, mask_token_index, :]

# 상위 5개 후보 뽑기 (topk)
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

print("\n🎯 예측 결과:")
for i, token_id in enumerate(top_5_tokens, 1):
    prediction = tokenizer.decode([token_id])
    print(f"  {i}. {prediction}")    # 3. Hub 모델 정보 확인하기
print("\n" + "=" * 60)
print("📋 모델 설정 정보")
print("=" * 60)

print(f"\n모델 이름: {checkpoint}")
print(f"히든 크기: {model.config.hidden_size}")
print(f"어텐션 헤드 수: {model.config.num_attention_heads}")
print(f"레이어 수: {model.config.num_hidden_layers}")
print(f"어휘 크기: {model.config.vocab_size:,}") from huggingface_hub import notebook_login

# Colab/Jupyter 환경에서 로그인
# 실행하면 토큰 입력 창이 나타납니다
# https://huggingface.co/settings/tokens 에서 토큰을 발급받으세요

print("=" * 60)
print("🔐 Hugging Face Hub 로그인")
print("=" * 60)
print("\n아래 셀을 실행하면 토큰 입력 창이 나타납니다.")
print("Hub 설정 페이지에서 'write' 권한이 있는 토큰을 발급받아 입력하세요.")
print("\n토큰 발급: https://huggingface.co/settings/tokens")    # 주석 해제 후 실행하세요
# notebook_login()    print("=" * 60)
print("🚀 Trainer를 이용한 Hub 업로드 설정")
print("=" * 60)

from transformers import TrainingArguments, Trainer

# Hub 업로드를 위한 TrainingArguments 예시
# ⚠️ 'YOUR_HF_USERNAME'을 자신의 허깅페이스 사용자명으로 변경하세요!

hub_training_args = TrainingArguments(
    output_dir="./nsmc-finetuned-bert",
    eval_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    # Hub 업로드 관련 설정
    push_to_hub=True,  # 훈련 완료 후 자동 업로드
    hub_model_id="YOUR_HF_USERNAME/nsmc-finetuned-bert",  # "사용자명/모델이름"
)

print("\n📋 Hub 업로드 설정:")
print(f"  - push_to_hub: {hub_training_args.push_to_hub}")
print(f"  - hub_model_id: {hub_training_args.hub_model_id}")
print(f"  - output_dir: {hub_training_args.output_dir}")

print("\n💡 Trainer로 훈련하면 자동으로 Hub에 업로드됩니다!")
print("   trainer = Trainer(model=model, args=hub_training_args, ...)")
print("   trainer.train()  # 훈련 완료 후 자동 업로드")    print("=" * 60)
print("🚀 push_to_hub() 메소드로 직접 업로드")
print("=" * 60)

# ⚠️ 'YOUR_HF_USERNAME'을 자신의 허깅페이스 사용자명으로 변경하세요!
repo_id = "YOUR_HF_USERNAME/nsmc-finetuned-bert-manual"

print(f"\n📦 저장소 ID: {repo_id}")
print("\n🔧 업로드 코드 예시:")
print("""
# 모델 로컬 저장
model.save_pretrained("./my-local-model")
tokenizer.save_pretrained("./my-local-model")

# Hub에 업로드
model.push_to_hub(repo_id)       # 모델 업로드
tokenizer.push_to_hub(repo_id)   # 토크나이저 업로드
""")

print("✅ 모델과 토크나이저를 각각 push_to_hub()로 업로드합니다.")
print("💡 같은 repo_id를 사용하면 같은 저장소에 함께 저장됩니다.")

# 실제 업로드 (주석 해제 후 실행)
# model.push_to_hub(repo_id)
# tokenizer.push_to_hub(repo_id)  
