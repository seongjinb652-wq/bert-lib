# bert-lib

00~09 한세트

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
