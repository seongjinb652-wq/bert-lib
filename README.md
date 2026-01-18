# bert-lib

# BERT 한국어 감성 분석 & 뉴스 토픽 분류 파이프라인

이 프로젝트는 Hugging Face Transformers와 KLUE/NSMC 데이터셋을 활용하여  
한국어 감성 분석 및 뉴스 토픽 분류 모델을 파인튜닝하고,  
Hub 업로드 및 모델 카드 작성까지 전체 과정을 정리한 파이프라인입니다.

---

## 📂 파일 구조 (00 ~ 99)

### 00~09: 기본 환경 및 설정
- **04_colab_korean_font_setup.py**  
  Colab 환경에서 Matplotlib 한글 폰트 설정 (NanumGothic 다운로드 및 적용).  
- **05_bert_finetune_pre_eval.py**  
  파인튜닝 전 모델 성능 평가 (베이스라인 확인).  
- **06_bert_finetune_config.py**  
  TrainingArguments 및 메트릭 정의.  
- **07_bert_finetune_train.py**  
  Trainer를 이용한 학습 실행.  
- **08_bert_finetune_metrics.py**  
  학습 후 메트릭 계산 (Accuracy, F1).  
- **09_bert_finetune_eval.py**  
  최종 평가 및 성능 확인.  

---

### 70~89: 데이터 준비 및 학습
- **79_nsmc_dataset_preprocess_tokenize.py**  
  NSMC 데이터셋 로드, 전처리, 토큰화, Trainer 입력 형식 변환.  
- **89_bert_sentiment_nsmc_finetune.py**  
  NSMC 감성 분석 모델 파인튜닝 및 학습 전후 성능 평가.  

---

### 90~97: 모델 저장 및 추가 데이터셋 준비
- **97_sentiment_and_topic_model_save_and_ynat.py**  
  NSMC 모델 저장 및 감성 분석 테스트, KLUE YNAT 데이터셋 로드 및 파인튜닝 준비.  

---

### 98~99: Hub 업로드 및 모델 카드
- **98_hub_upload.py**  
  Hugging Face Hub 업로드 (Trainer 자동 업로드 및 push_to_hub 수동 업로드 예시).  
- **99_model_card.py**  
  모델 카드(README.md) 템플릿 생성 및 저장.  

---

## 🚀 실행 순서

1. **데이터 준비**  
   - 79 → NSMC 데이터셋 전처리 및 토큰화  
2. **모델 학습**  
   - 89 → NSMC 감성 분석 모델 파인튜닝  
3. **모델 저장 및 추가 데이터셋 준비**  
   - 97 → 모델 저장 및 YNAT 데이터셋 준비  
4. **Hub 업로드**  
   - 98 → Hugging Face Hub 업로드  
5. **모델 카드 작성**  
   - 99 → 모델 카드 생성 및 저장  

---

## 📊 결과 요약
- **NSMC 감성 분석**  
  - Accuracy: ~0.89  
  - F1 Score: ~0.89  
- **YNAT 뉴스 토픽 분류**  
  - 7개 클래스 분류 준비 완료 (실행 시 Trainer로 학습 가능).  

---

## 📌 참고
- Hugging Face Hub: [https://huggingface.co](https://huggingface.co)  
- NSMC 데이터셋: [https://github.com/e9t/nsmc](https://github.com/e9t/nsmc)  
- KLUE YNAT 데이터셋: [https://huggingface.co/datasets/klue](https://huggingface.co/datasets/klue)  


00~09 한세트

토큰 발급 절차:

""" bash
huggingface.co/settings/tokens 접속
"New token" 클릭
이름 입력 및 권한 선택 (write 권한 필요)
토큰 복사 및 안전하게 보관
"""

""" 
Library: KoNLP-Finetune
Author: 성진
Date: 2026-01-18

Description:
    한국어 자연어 처리(NLP) 모델 파인튜닝 및 평가를 위한 라이브러리.
    Hugging Face Transformers 기반으로 BERT 등 사전학습 모델을 불러와
    텍스트 분류 태스크에 맞게 학습, 평가, 파이프라인 실행을 지원합니다.

Features:
    - 데이터 전처리 및 토크나이저 설정
    - 학습/검증 데이터셋 구성
    - Trainer 기반 학습 및 평가
    - Accuracy, F1 등 주요 메트릭 계산
    - 파인튜닝 전/후 성능 비교

Dependencies:
    - transformers
    - datasets
    - evaluate
    - numpy
    - matplotlib (옵션: 시각화)

Usage:
    from konlp_finetune import TrainerPipeline

    pipeline = TrainerPipeline(model="bert-base-multilingual-cased")
    pipeline.train(train_data, val_data)
    pipeline.evaluate(test_data)
"""
