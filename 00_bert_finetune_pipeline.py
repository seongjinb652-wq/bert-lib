"""
File: bert_finetune_pipeline.py
Author: 성진
Date: 2026-01-18

Description:
    BERT 텍스트 분류 모델 파인튜닝 전체 과정을 순차적으로 실행하는 파이프라인 스크립트입니다.
    개별 단계별 스크립트(04~09)를 차례대로 호출하여 학습 환경 설정, 
    파인튜닝 전 평가, 학습 설정, 학습 실행, 메트릭 계산, 최종 평가를 자동으로 수행합니다.

Steps:
    1. 04_colab_korean_font_setup.py   → Colab 환경 한글 폰트 설정
    2. 05_bert_finetune_pre_eval.py    → 파인튜닝 전 모델 평가 (베이스라인)
    3. 06_bert_finetune_config.py      → 학습 설정 및 메트릭 정의
    4. 07_bert_finetune_train.py       → 모델 학습 실행
    5. 08_bert_finetune_metrics.py     → 학습 후 메트릭 계산
    6. 09_bert_finetune_eval.py        → 최종 평가 및 성능 확인

Dependencies:
    - Python 3.x
    - transformers, datasets, evaluate, matplotlib, numpy
    - 개별 단계별 스크립트 파일 (04~09)

Usage:
    $ python bert_finetune_pipeline.py

Note:
    - 각 단계는 독립적으로 실행 가능하지만, 전체 파이프라인 실행 시 순차적으로 호출됩니다.
    - Colab 환경에서는 04번 스크립트 실행 시 wget 명령으로 폰트 다운로드 필요.
"""
import subprocess

# 실행할 스크립트 목록
scripts = [
    "04_colab_korean_font_setup.py",
    "05_bert_finetune_pre_eval.py",
    "06_bert_finetune_config.py",
    "07_bert_finetune_train.py",
    "08_bert_finetune_metrics.py",
    "09_bert_finetune_eval.py",
]

def run_pipeline():
    print("=" * 60)
    print("🚀 BERT 파인튜닝 전체 파이프라인 실행 시작")
    print("=" * 60)

    for script in scripts:
        print(f"\n▶ 실행 중: {script}")
        try:
            subprocess.run(["python", script], check=True)
            print(f"✅ 완료: {script}")
        except subprocess.CalledProcessError:
            print(f"❌ 오류 발생: {script}")
            break

    print("\n🎉 전체 파이프라인 실행 완료!")

if __name__ == "__main__":
    run_pipeline()

