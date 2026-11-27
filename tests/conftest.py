import sys
from pathlib import Path

# 패키지를 설치하지 않고 src/ 아래 모듈을 import한다
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
