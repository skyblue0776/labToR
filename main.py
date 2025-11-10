# main.py
# import pathlib
#
# APP_PATH = str(pathlib.Path(__file__).resolve())

# 1) 런처: 직접 실행이면 streamlit run으로 재실행 후 종료
# if __name__ == "__main__" and os.getenv("RUNNING_IN_STREAMLIT") != "1":
#     env = os.environ.copy()
#     env["RUNNING_IN_STREAMLIT"] = "1"  # 스트림릿 실행 컨텍스트 표시
#     # 필요하면 --server.headless true 등 옵션 추가 가능
#     subprocess.run([
#         sys.executable, "-m", "streamlit", "run", APP_PATH,
#         "--server.port", "8501",            # 여기만 바꾸면 끝
#     ], env=env)
#     sys.exit(0)

# 2) 여기부터가 실제 Streamlit 앱 코드 (streamlit import는 런처 분기 뒤!)
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st
import streamlit.components.v1 as components
