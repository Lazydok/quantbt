import os
from pathlib import Path

# 이 파일(config.py)의 위치를 기준으로 프로젝트 루트를 동적으로 결정합니다.
# config.py -> quantbt -> project_root
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent

PROJECT_ROOT = str(PROJECT_ROOT_PATH)
DATA_DIR = str(PROJECT_ROOT_PATH / "data")
CACHE_DIR = str(PROJECT_ROOT_PATH / "data" / "cache")
DB_PATH = str(PROJECT_ROOT_PATH / "data" / "quantbt.db")

# 데이터 디렉토리와 캐시 디렉토리가 없으면 생성합니다.
os.makedirs(CACHE_DIR, exist_ok=True)