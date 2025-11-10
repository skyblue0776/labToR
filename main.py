# main.py
import os
import sys
import subprocess
import pathlib

APP_PATH = str(pathlib.Path(__file__).resolve())

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

# ---------------------- 기본 설정 ----------------------
st.set_page_config(page_title="Reflectance Generator", page_icon="🎨", layout="centered")
st.title("Reflectance Generator")

WAVELENGTHS_31 = list(range(400, 701, 10))  # 400~700nm, 10nm 간격 → 31개
WAVELENGTHS_ALL = list(range(360, 781, 10))  # 360~780nm, 10nm 간격 → 43개

st.caption(
    "1. L*, a*, b* 입력\n"
    "2. 반사율 만들기\n"
    "3. 클립보드로 복사"
)


# ---------------------- 데이터 로드 ----------------------
@st.cache_data(show_spinner=False)
def load_xyz(path: str = "xyz.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # 400~700nm(31개)에 해당하는 행만 상단 31행으로 사용한다고 가정
    # (WL 컬럼이 있으면 그걸로 정렬/필터)
    if "WL" in df.columns:
        df = df.sort_values("WL")
        df = df[df["WL"].isin(WAVELENGTHS_31)]
        if len(df) < 31:
            raise ValueError("xyz.csv에 400~700nm(10nm 간격) 31행이 부족합니다.")
        df = df.iloc[:31].reset_index(drop=True)
    else:
        if len(df) < 31:
            raise ValueError("xyz.csv 행 개수가 31보다 적습니다. 400~700nm 31행이 필요합니다.")
        df = df.iloc[:31].reset_index(drop=True)

    needed = ["Wx(D65-10)", "Wy(D65-10)", "Wz(D65-10)",
              "Wx(F02)", "Wy(F02)", "Wz(F02)",
              "Wx(A-10)", "Wy(A-10)", "Wz(A-10)"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"xyz.csv에 다음 컬럼이 없습니다: {miss}")
    return df


try:
    XYZ = load_xyz("xyz.csv")
except Exception as e:
    st.error(f"xyz.csv 로드 실패: {e}")
    st.stop()


# ---------------------- Lab 변환 유틸 ----------------------
def _f_piecewise(t):
    t1 = (6 / 29) ** 3
    t2 = (29 / 6) ** 2 / 3
    return np.where(t > t1, np.cbrt(t), t2 * t + 16 / 116)


def xyz_to_lab(x, y, z, white_xyz):
    Xn, Yn, Zn = white_xyz
    Xr, Yr, Zr = x / Xn, y / Yn, z / Zn
    fx, fy, fz = _f_piecewise(Xr), _f_piecewise(Yr), _f_piecewise(Zr)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return L, a, b


def reflectance_to_lab(reflectance_01: pd.DataFrame, XYZ: pd.DataFrame) -> pd.DataFrame:
    """
    reflectance_01: (31 x N) DataFrame, 0~1 반사율
    반환: 9 x N (D65/F02/A 각각 L,a,b)
    """
    wx_d65 = XYZ["Wx(D65-10)"].values
    wy_d65 = XYZ["Wy(D65-10)"].values
    wz_d65 = XYZ["Wz(D65-10)"].values
    wx_f02 = XYZ["Wx(F02)"].values
    wy_f02 = XYZ["Wy(F02)"].values
    wz_f02 = XYZ["Wz(F02)"].values
    wx_a10 = XYZ["Wx(A-10)"].values
    wy_a10 = XYZ["Wy(A-10)"].values
    wz_a10 = XYZ["Wz(A-10)"].values

    white_d65 = (94.813, 99.997, 107.304)
    white_f02 = (103.281, 100.002, 69.03)
    white_a10 = (111.143, 99.999, 35.201)

    n_cols = reflectance_01.shape[1]
    out = pd.DataFrame(
        np.ones((9, n_cols)),
        index=["L*(D65)", "a*(D65)", "b*(D65)", "L*(F02)", "a*(F02)", "b*(F02)", "L*(A)", "a*(A)", "b*(A)"],
        columns=reflectance_01.columns,
    )

    for i in range(n_cols):
        r = reflectance_01.iloc[:, i].values  # 31,
        Xd = np.sum(r * wx_d65);
        Yd = np.sum(r * wy_d65);
        Zd = np.sum(r * wz_d65)
        Xf = np.sum(r * wx_f02);
        Yf = np.sum(r * wy_f02);
        Zf = np.sum(r * wz_f02)
        Xa = np.sum(r * wx_a10);
        Ya = np.sum(r * wy_a10);
        Za = np.sum(r * wz_a10)
        Ld, ad, bd = xyz_to_lab(Xd, Yd, Zd, white_d65)
        Lf, af, bf = xyz_to_lab(Xf, Yf, Zf, white_f02)
        La, aa, ba = xyz_to_lab(Xa, Ya, Za, white_a10)
        out.iloc[:, i] = [Ld, ad, bd, Lf, af, bf, La, aa, ba]
    return out


# ---------------------- 최적화 ----------------------
def solve_reflectance_for_lab(L_target: float, a_target: float, b_target: float, XYZ: pd.DataFrame):
    """
    목표 Lab(D65)에 가장 근접하도록 31-포인트 반사율(0~1) 최적화 (SLSQP).
    반환: OptimizeResult, reflectance_01(31,)
    """
    x0 = np.full(31, 0.5, dtype=float)
    bounds = [(0.0, 1.0)] * 31

    def objective(x):
        df_r = pd.DataFrame(x, columns=["r"])  # (31 x 1)
        lab = reflectance_to_lab(df_r, XYZ).T  # (1 x 9)
        Ld, ad, bd = lab.iloc[0, 0], lab.iloc[0, 1], lab.iloc[0, 2]
        return (Ld - L_target) ** 2 + (ad - a_target) ** 2 + (bd - b_target) ** 2

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-8, "disp": False})
    return res, np.clip(res.x, 0.0, 1.0)


# ---------------------- 클립보드 복사 ----------------------
def copy_text_to_clipboard(text: str, ok_msg: str, fail_msg: str):
    """
    Streamlit 브라우저(프론트) 클립보드 복사. 화면에 아무 CSV/TSV도 표시/다운로드하지 않음.
    """
    safe_js = json.dumps(text)  # JS로 안전 전달
    components.html(
        f"""
        <script>
          (function() {{
            const data = {safe_js};
            navigator.clipboard.writeText(data)
              .then(() => {{
                const d = document.createElement('div');
                d.textContent = {json.dumps(ok_msg)};
                d.style.cssText='padding:8px 10px;margin:6px 0;background:#eef;border:1px solid #99f;border-radius:8px;';
                document.currentScript.parentElement.appendChild(d);
              }})
              .catch(e => {{
                const d = document.createElement('div');
                d.textContent = {json.dumps(fail_msg)} + ' ' + e;
                d.style.cssText='color:#b00;margin:6px 0;';
                document.currentScript.parentElement.appendChild(d);
              }});
          }})();
        </script>
        """,
        height=0
    )


# ---------------------- DataFrame 구성 ----------------------
def build_full_dataframe(reflectance_400_700_percent):
    """
    400~700nm(31개, %) 반사율을 받아 메타데이터 + 360~780nm 전체(43개) 컬럼을 포함한 1행 DataFrame 생성.
    400~700nm 구간은 계산값으로 덮어쓰고, 나머지(360~390, 710~780)는 기본값(여기선 100)으로 채움.
    """
    data = {
        "Name": ["W1"],
        "DateTime": ["2023-09-19 오후 2:45:35"],
        "MeasCond": ["R4IMPNNp6IA1"],
        "StdBat": ["B"],
        "Comment": [""],
        "UUID": ["0Y3uRCVM"],
        "eUUID": [-695076295],
        "GUID": ["75335930-4D56-4352-D691-FA39D691FA39"],
        "L*": [100],
        "a*": [0],
        "b*": [0],
        "Conc": [0],
        "SWL": [400],
        "EWL": [700],
    }

    # 기본값(예: 100)로 360~780 채우고, 400~700만 계산값으로 대체
    DEFAULT_VALS = [100.0] * len(WAVELENGTHS_ALL)
    idx_map_31 = {wl: i for i, wl in enumerate(WAVELENGTHS_31)}

    for i, wl in enumerate(WAVELENGTHS_ALL):
        col = str(wl)
        if wl in idx_map_31:
            data[col] = [float(reflectance_400_700_percent[idx_map_31[wl]])]
        else:
            data[col] = [float(DEFAULT_VALS[i])]

    return pd.DataFrame(data)


# ---------------------- UI ----------------------
st.subheader("입력값")
c1, c2, c3 = st.columns(3)
L_in = c1.number_input("L*", value=50.0, step=0.1, format="%.3f")
a_in = c2.number_input("a*", value=20.0, step=0.1, format="%.3f")
b_in = c3.number_input("b*", value=-10.0, step=0.1, format="%.3f")
go = st.button("반사율 만들기", use_container_width=True)
if go:
    with st.spinner("최적화 중…"):
        res, r01 = solve_reflectance_for_lab(L_in, a_in, b_in, XYZ)
    if not res.success:
        st.warning(f"최적화 수렴 경고: status={res.status}, message={res.message}")

    # 0~100% 변환
    r_percent = r01 * 100.0

    # 전체 DataFrame(메타데이터 + 360~780nm) 구성
    df_full = build_full_dataframe(r_percent)

    # CSV 문자열로 직렬화 (표시/다운로드 없이, 클립보드에만 넣기)
    tsv_str = df_full.to_csv(index=False, sep="\t", header=True)  # ← 탭 구분!
    print(tsv_str)
    data_js = json.dumps(tsv_str)  # JS로 안전 전달
    components.html(
        f"""
            <div id="copy-root" style="display:flex; gap:8px; align-items:center;">
              <button id="copy-btn" style="
                padding:8px 12px;border-radius:8px;border:1px solid #ccc;
                cursor:pointer;background:#f0f2f6;">{"클립보드로 복사"}</button>
              <span id="copy-msg" style="font-family:system-ui,Arial; font-size:14px;"></span>
            </div>
            <textarea id="copy-ta" style="position:absolute; left:-9999px; top:-9999px;"></textarea>

            <script>
              (function () {{
                const btn = document.getElementById('copy-btn');
                const msg = document.getElementById('copy-msg');
                const ta  = document.getElementById('copy-ta');
                const text = {data_js};  // TSV 문자열

                function setMsg(ok, detail="") {{
                  if (ok) {{
                    msg.textContent = "✅ 복사 완료";
                    msg.style.color = "#0a0";
                  }} else {{
                    msg.textContent = "❌ 복사 실패 " + (detail ? "(" + detail + ")" : "");
                    msg.style.color = "#b00";
                  }}
                }}

                async function copyWithClipboardAPI() {{
                  await navigator.clipboard.writeText(text);
                }}

                function copyWithExecCommand() {{
                  ta.value = text;
                  ta.select();
                  ta.setSelectionRange(0, ta.value.length);
                  const ok = document.execCommand('copy');
                  return ok;
                }}

                btn.addEventListener('click', async () => {{
                  try {{
                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                      await copyWithClipboardAPI();
                      setMsg(true);
                      return;
                    }}
                  }} catch (e) {{}}

                  try {{
                    const ok = copyWithExecCommand();
                    setMsg(ok, ok ? "" : "execCommand 실패");
                  }} catch (e) {{
                    setMsg(false, e);
                  }}
                }});
              }})();
            </script>
            """,
        height=40,
    )
    st.success("클립보드 복사 완료")

c21, c22, c23 = st.columns(3)
L2_in = c21.number_input("L*", value=50.0, step=0.1, format="%.3f", key="L2")
a2_in = c22.number_input("a*", value=20.0, step=0.1, format="%.3f", key="a2")
b2_in = c23.number_input("b*", value=-10.0, step=0.1, format="%.3f", key="b2")
go2 = st.button("반사율 만들기", use_container_width=True, key="bt2")
if go2:
    with st.spinner("최적화 중…"):
        res, r01 = solve_reflectance_for_lab(L2_in, a2_in, b2_in, XYZ)
    if not res.success:
        st.warning(f"최적화 수렴 경고: status={res.status}, message={res.message}")

    # 0~100% 변환
    r_percent = r01 * 100.0

    # 전체 DataFrame(메타데이터 + 360~780nm) 구성
    df_full = build_full_dataframe(r_percent)

    # CSV 문자열로 직렬화 (표시/다운로드 없이, 클립보드에만 넣기)
    tsv_str = df_full.to_csv(index=False, sep="\t", header=True)  # ← 탭 구분!
    data_js = json.dumps(tsv_str)  # JS로 안전 전달
    components.html(
        f"""
            <div id="copy-root" style="display:flex; gap:8px; align-items:center;">
              <button id="copy-btn" style="
                padding:8px 12px;border-radius:8px;border:1px solid #ccc;
                cursor:pointer;background:#f0f2f6;">{"클립보드로 복사"}</button>
              <span id="copy-msg" style="font-family:system-ui,Arial; font-size:14px;"></span>
            </div>
            <textarea id="copy-ta" style="position:absolute; left:-9999px; top:-9999px;"></textarea>

            <script>
              (function () {{
                const btn = document.getElementById('copy-btn');
                const msg = document.getElementById('copy-msg');
                const ta  = document.getElementById('copy-ta');
                const text = {data_js};  // TSV 문자열

                function setMsg(ok, detail="") {{
                  if (ok) {{
                    msg.textContent = "✅ 복사 완료";
                    msg.style.color = "#0a0";
                  }} else {{
                    msg.textContent = "❌ 복사 실패 " + (detail ? "(" + detail + ")" : "");
                    msg.style.color = "#b00";
                  }}
                }}

                async function copyWithClipboardAPI() {{
                  await navigator.clipboard.writeText(text);
                }}

                function copyWithExecCommand() {{
                  ta.value = text;
                  ta.select();
                  ta.setSelectionRange(0, ta.value.length);
                  const ok = document.execCommand('copy');
                  return ok;
                }}

                btn.addEventListener('click', async () => {{
                  try {{
                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                      await copyWithClipboardAPI();
                      setMsg(true);
                      return;
                    }}
                  }} catch (e) {{}}

                  try {{
                    const ok = copyWithExecCommand();
                    setMsg(ok, ok ? "" : "execCommand 실패");
                  }} catch (e) {{
                    setMsg(false, e);
                  }}
                }});
              }})();
            </script>
            """,
        height=40,
    )
    st.success("클립보드 복사 완료")
