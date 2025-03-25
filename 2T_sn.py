import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. CSV 파일 읽기 (바탕화면에 "2T0C.csv"가 있다고 가정)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "2T0C.csv")
print("CSV 파일 경로:", desktop_path)
data = pd.read_csv(desktop_path)

# 2. V_GS가 같은 데이터끼리 그룹화하여 저장
# 각 그룹은 V_DS에 대해 오름차순 정렬된 (V_DS, I_D) 데이터를 가짐
groups = {}
unique_vgs = np.sort(data["V_GS"].unique())
for vgs in unique_vgs:
    subset = data[data["V_GS"] == vgs].sort_values(by="V_DS")
    groups[vgs] = {
        "V_DS": subset["V_DS"].to_numpy(),
        "I_D": subset["I_D"].to_numpy()
    }

# 3. 주어진 V_GS에 대해, 해당 그룹이 없으면 가장 가까운 그룹을 선택하는 함수
def get_group(vgs_target):
    if vgs_target in groups:
        return groups[vgs_target]
    all_vgs = np.array(list(groups.keys()))
    idx = (np.abs(all_vgs - vgs_target)).argmin()
    closest = all_vgs[idx]
    return groups[closest]

# 4. 초기값 설정
# 원래 곡선의 V_GS (Vgs1)는 사용자 입력, 대칭 곡선의 V_GS (Vgs2)는 0.1로 설정
Vgs1 = float(input("원래 곡선의 초기 V_GS (Vgs1) [V]를 입력하세요: "))
Vgs2 = 0.1

# --- (1) Vgs1 데이터 불러오기 ---
group1 = get_group(Vgs1)
v1 = group1["V_DS"]
i1 = group1["I_D"]

# --- (2) Vgs2 (=0.1) 데이터 불러오기 ---
group2 = get_group(Vgs2)
v2_original = group2["V_DS"]
i2_original = group2["I_D"]

# --- (3) Vgs2 쪽에 x=1에 대한 대칭 적용 ---
#     x' = 2 - x
v2_reflected = 2.0 - v2_original
i2_reflected = i2_original  # I_D는 그대로 두고, V_DS만 x=1에 대해 반사

# (참고) 정렬(오름차순)해서 보간 시 안정적으로 사용하기
order = np.argsort(v2_reflected)
v2_reflected = v2_reflected[order]
i2_reflected = i2_reflected[order]

# --- (4) 보간(interpolation)을 위한 공통 V_DS 축 구성 ---
# 두 배열의 유효 구간이 겹치는 부분만 확인하면 되므로
v_min = max(v1.min(), v2_reflected.min())
v_max = min(v1.max(), v2_reflected.max())

# 예: 1000개 지점으로 균등 분할
v_common = np.linspace(v_min, v_max, 1000)

# 보간: np.interp(x, xp, fp)는 xp, fp를 이용해 x에서의 값을 선형 보간
i1_interp = np.interp(v_common, v1, i1)
i2_interp = np.interp(v_common, v2_reflected, i2_reflected)

# --- (5) 두 곡선의 교점 근사: 부호 변화 지점만 확인 ---
diff = i1_interp - i2_interp
sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]  # 부호가 바뀌는 인덱스

if len(sign_changes) == 0:
    print("두 곡선 사이에 교점을 찾지 못했습니다.")
else:
    # 여러 교점이 있을 수 있으니 모두 확인 가능
    # 여기서는 각 부호변화 구간에 대해 '직전'과 '직후' 정보를 보여줌
    for idx in sign_changes:
        # diff[idx]와 diff[idx+1] 사이에서 부호가 바뀐다.
        v_before = v_common[idx]
        v_after  = v_common[idx+1]
        i1_before = i1_interp[idx]
        i1_after  = i1_interp[idx+1]
        i2_before = i2_interp[idx]
        i2_after  = i2_interp[idx+1]

        print(f"\n=== 부호 변화 지점(index={idx}) ===")
        print(f"[직전] V_DS={v_before:.4f}, i1={i1_before:.6g}, i2={i2_before:.6g}, diff={diff[idx]:.6g}")
        print(f"[직후]  V_DS={v_after:.4f}, i1={i1_after:.6g},  i2={i2_after:.6g},  diff={diff[idx+1]:.6g}")

    # (선택) 가장 앞선 교점 하나만 그래프에 표시
    idx_first = sign_changes[0]
    # 교점 근사 좌표(직전-직후 중간쯤)를 그냥 시각화 용도로 잡아봄
    x_cross_approx = 0.5*(v_common[idx_first] + v_common[idx_first+1])
    i1_cross_approx = 0.5*(i1_interp[idx_first] + i1_interp[idx_first+1])

    # --- (6) 그래프 시각화 ---
    plt.figure(figsize=(8,6))
    # 원래 곡선
    plt.plot(v1, i1, 'b-o', label=f"Original: Vgs={Vgs1} V")
    # 대칭 곡선
    plt.plot(v2_reflected, i2_reflected, 'r-o', label=f"Reflected: Vgs={Vgs2} V (x->2-x)")

    # 부호 바뀌는 지점 근사값(첫 번째만)
    plt.plot(x_cross_approx, i1_cross_approx, 'ko', label="First sign-change region")

    plt.xlabel("V_DS (V)")
    plt.ylabel("I_D (A)")
    plt.legend()
    plt.grid(True)
    plt.title("교점(부호변화) 직전/직후 데이터 확인 예시")
    plt.show()
