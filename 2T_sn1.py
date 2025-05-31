import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. CSV 파일 읽기 (바탕화면 "2T0C.csv" 가정)
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "2T0C.csv")
data = pd.read_csv(desktop_path)

# 2. V_GS별로 그룹화
groups = {}
unique_vgs = np.sort(data["V_GS"].unique())
for vgs in unique_vgs:
    subset = data[data["V_GS"] == vgs].sort_values(by="V_DS")
    groups[vgs] = {
        "V_DS": subset["V_DS"].to_numpy(), 
        "I_D": subset["I_D"].to_numpy()
    }

# 3. 주어진 V_GS에 대해 (정확 일치 X 시) 가장 가까운 그룹
def get_group(vgs_target):
    if vgs_target in groups:
        return groups[vgs_target]
    all_vgs = np.array(list(groups.keys()))
    idx = (np.abs(all_vgs - vgs_target)).argmin()
    return groups[all_vgs[idx]]

# 4. 사용자 입력
Vgs1 = float(input("원래 곡선의 초기 V_GS (Vgs1) [V]를 입력하세요: "))
Vgs2 = 2.0

# --- (A) Vgs1 그룹 ---
grp1 = get_group(Vgs1)
v1 = grp1["V_DS"]
i1 = grp1["I_D"]

# --- (B) Vgs2 그룹 ---
grp2 = get_group(Vgs2)
v2_orig = grp2["V_DS"]
i2_orig = grp2["I_D"]

# --- (C) x=1 대칭 ---
v2_refl = 2.0 - v2_orig
i2_refl = i2_orig

# (C-1) 정렬
idx_sort = np.argsort(v2_refl)
v2_refl = v2_refl[idx_sort]
i2_refl = i2_refl[idx_sort]

# --- (D) "동일한 V_DS" 찾기 (보간없이) ---
i_idx = 0
j_idx = 0
matched = []  # (V_DS, I1, I2, diff)

while i_idx < len(v1) and j_idx < len(v2_refl):
    if v1[i_idx] == v2_refl[j_idx]:
        diff_ij = i1[i_idx] - i2_refl[j_idx]
        matched.append((v1[i_idx], i1[i_idx], i2_refl[j_idx], diff_ij))
        i_idx += 1
        j_idx += 1
    elif v1[i_idx] < v2_refl[j_idx]:
        i_idx += 1
    else:
        j_idx += 1

# (E) 부호가 바뀌는 구간 찾기
sign_changes = []
for k in range(len(matched) - 1):
    diff_k  = matched[k][3]
    diff_k1 = matched[k+1][3]
    # (단순) 부호 변화
    if diff_k * diff_k1 < 0:
        sign_changes.append(k)

if not sign_changes:
    print("두 곡선 사이에 (동일 V_DS 상) 부호가 바뀌는 구간을 찾지 못했습니다.")
else:
    for idx in sign_changes:
        # 직전 점
        v_before, i1_bef, i2_bef, diff_bef = matched[idx]
        # 직후 점
        v_after,  i1_aft, i2_aft, diff_aft = matched[idx+1]

        print("\n=== 부호 변화 지점 (인덱스: {}→{}) ===".format(idx, idx+1))
        print("[직전] V_DS={:.4f}, I1={:.6g}, I2={:.6g}, diff={:.6g}"
              .format(v_before, i1_bef, i2_bef, diff_bef))
        print("[직후] V_DS={:.4f}, I1={:.6g}, I2={:.6g}, diff={:.6g}"
              .format(v_after, i1_aft, i2_aft, diff_aft))

        # -----------------------
        # (F) 선분 교차점(교점) 계산
        # -----------------------
        # Line1 (원래 곡선): (v_before, i1_bef) ~ (v_after, i1_aft)
        # Line2 (대칭 곡선): (v_before, i2_bef) ~ (v_after, i2_aft)
        #
        # A = (i1_bef - i2_bef)
        # B = (i1_aft - i1_bef) - (i2_aft - i2_bef)
        #    => A + B*t = 0 --> t = -A/B
        #
        A = i1_bef - i2_bef
        B = (i1_aft - i1_bef) - (i2_aft - i2_bef)

        # 예외처리 (B=0 -> 평행 혹은 중복)
        if abs(B) < 1e-15:
            if abs(A) < 1e-15:
                print(" -> 이 구간에서 두 선분이 거의 일치(무수히 많은 교점) 혹은 diff=0인 상태입니다.")
            else:
                print(" -> 이 구간에서 두 선분이 평행하고 교차점이 없습니다.")
        else:
            t_star = -A / B
            # 실제 교점이 선분 안쪽에 있는지 확인 (0 <= t_star <= 1)
            if 0 <= t_star <= 1:
                x_star = v_before + (v_after - v_before)*t_star
                # y_star: 원래 곡선의 값(또는 대칭 곡선도 동일)
                y_star_1 = i1_bef + (i1_aft - i1_bef)*t_star
                # (확인용) y_star_2 = i2_bef + (i2_aft - i2_bef)*t_star

                print(" -> 교점(선분 내부)에 대한 t = {:.4f}".format(t_star))
                print(" -> 교점 좌표: V_DS = {:.4f}, I_D = {:.6g}".format(x_star, y_star_1))
            else:
                print(" -> 교점이 선분 범위 밖에 위치 (t={:.4f}), 교점이 구간 내부에 없음.".format(t_star))


# (선택) 그래프 시각화
if matched:
    v_m = np.array([m[0] for m in matched])
    i1_m = np.array([m[1] for m in matched])
    i2_m = np.array([m[2] for m in matched])

    plt.figure(figsize=(8,6))
    plt.plot(v1, i1, 'bo-', label=f"원래 곡선 (Vgs={Vgs1} V)")
    plt.plot(v2_refl, i2_refl, 'ro-', label=f"대칭 곡선 (Vgs={Vgs2} V, x→2-x)")
    # matched 지점
    plt.plot(v_m, i1_m, 'bs', label="매칭점(I1)")
    plt.plot(v_m, i2_m, 'rs', label="매칭점(I2)")

    plt.xlabel("V_DS (V)")
    plt.ylabel("I_D (A)")
    plt.title("부호변화 구간에서 선분 교차점 찾기 예시")
    plt.grid(True)
    plt.legend()
    plt.show()
