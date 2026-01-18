"""
File: colab_korean_font_setup.py
Author: 성진
Date: 2026-01-18

Description:
    Google Colab 환경에서 Matplotlib 그래프에 한글을 정상적으로 표시하기 위한
    폰트 설정 스크립트입니다. 나눔고딕(NanumGothic) 폰트를 다운로드 및 등록하여
    그래프 내 한글 깨짐 문제를 해결합니다.

Features:
    - NanumGothic-Regular.ttf 폰트 다운로드 (Colab 환경)
    - Matplotlib에 폰트 등록 및 적용
    - 마이너스 기호 깨짐 방지 설정

Dependencies:
    - matplotlib
    - matplotlib.font_manager
    - wget (Colab 환경에서 폰트 다운로드 시 사용)

Usage:
    1. Colab 환경에서 본 스크립트를 실행
    2. Matplotlib 그래프에 한글이 정상적으로 표시됨
    3. 필요 시 다른 한글 폰트로 교체 가능

Note:
    - 로컬 환경에서는 해당 폰트 파일을 직접 설치 후 경로를 지정해야 함
"""
# 환경 설정: 한글 폰트 (Colab 환경)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ------------------------------------------------------------
# 🔧 한글 폰트 설정 (Colab용)
# ------------------------------------------------------------
# Colab에서는 아래 주석을 해제하여 폰트 다운로드 필요
!wget 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf' -O 'NanumGothic.ttf'

# 폰트 파일이 있는 경우 등록
try:
    fm.fontManager.addfont("NanumGothic.ttf")
    plt.rc("font", family="NanumGothic")
except:
    print("⚠️ 한글 폰트 파일이 없습니다. Colab에서 wget 명령으로 다운로드하세요.")

# 마이너스 기호 깨짐 방지
plt.rc("axes", unicode_minus=False)
