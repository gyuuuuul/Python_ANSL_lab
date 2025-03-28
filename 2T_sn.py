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

# 3. V_GS에 대해 가장 가까운 그룹 가져오기
def get_group(vgs_target):
    if vgs_target in groups:
        return groups[vgs_target]
    all_vgs = np.array(list(groups.keys()))
    idx = (np.abs(all_vgs - vgs_target)).argmin()
    return groups[all_vgs[idx]]

# ------------------------------
# 교차점(선분 교차) 구하는 보조함수
# ------------------------------
def find_intersection(v1_before, v1_after, i1_before, i1_after,
                      v2_before, v2_after, i2_before, i2_after,
                      eps=1e-15):
    """
    원래 곡선 선분: (v1_before, i1_before) ~ (v1_after, i1_after)
    대칭 곡선 선분: (v2_before, i2_before) ~ (v2_after, i2_after)
    """
    A = (i1_before - i2_before)
    B = ((i1_after - i1_before) - (i2_after - i2_before))

    if abs(B) < eps:
        # 평행 또는 동일
        if abs(A) < eps:
            # 완전히 동일(무수히 많은 교점)
            return None
        else:
            # 평행 교점 없음
            return None
    else:
        t_star = -A / B
        # 선분 내부 교차인지 확인 (0 <= t <= 1)
        if 0 <= t_star <= 1:
            x_star = v1_before + (v1_after - v1_before)*t_star
            y_star = i1_before + (i1_after - i1_before)*t_star
            return (x_star, y_star)
        else:
            # 교점이 선분 범위 밖
            return None

# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":

    # 사용자 입력
    Vgs1 = float(input("Vgs1 [V]를 입력하세요: "))  # 예: 1.2
    Vgs2 = float(input("Vgs2 [V]를 입력하세요: "))  # 예: 2.0

    max_iter = 50
    tolerance = 1e-6  # V_DS 매칭 시 허용 오차

    # 고정 Vgs1 곡선 (원래 곡선)
    group1 = get_group(Vgs1)
    v1_all = group1["V_DS"]
    i1_all = group1["I_D"]

    # --- [추가] 각 iteration에서 그릴 데이터 보관 ---
    # 예: [
    #   {
    #     "iter": i,
    #     "Vgs2": (현재 Vgs2),
    #     "v2_refl": [...],
    #     "i2_refl": [...],
    #     "x_star": (교점 x),
    #     "y_star": (교점 y),
    #   }, ...
    # ]
    iteration_data = []

    found_condition = False

    for i in range(1, max_iter+1):
        # (1) Vgs2 그룹
        group2 = get_group(Vgs2)
        v2_orig = group2["V_DS"]
        i2_orig = group2["I_D"]

        # (2) x=1 대칭
        v2_refl = 2.0 - v2_orig
        i2_refl = i2_orig
        # 정렬
        idx_sort = np.argsort(v2_refl)
        v2_refl = v2_refl[idx_sort]
        i2_refl = i2_refl[idx_sort]

        # (3) 보간 없이 매칭
        matched = []
        p1, p2 = 0, 0
        while p1 < len(v1_all) and p2 < len(v2_refl):
            if np.isclose(v1_all[p1], v2_refl[p2], atol=tolerance):
                diff_ij = i1_all[p1] - i2_refl[p2]
                matched.append((v1_all[p1], i1_all[p1], i2_refl[p2], diff_ij))
                p1 += 1
                p2 += 1
            elif v1_all[p1] < v2_refl[p2]:
                p1 += 1
            else:
                p2 += 1

        if len(matched) < 2:
            print(f"\n반복 {i}: Vgs2={Vgs2:.4f}")
            print("  매칭점이 2개 미만 => 교점 계산 불가. 반복 종료.")
            break

        # (4) 부호변화 지점 찾기
        sign_change_indices = []
        for k in range(len(matched)-1):
            diff_k  = matched[k][3]
            diff_k1 = matched[k+1][3]
            s_k  = np.sign(diff_k)
            s_k1 = np.sign(diff_k1)
            if s_k != s_k1:
                sign_change_indices.append(k)

        print(f"\n반복 {i}: Vgs2={Vgs2:.4f}, 부호변화 개수={len(sign_change_indices)}")

        if not sign_change_indices:
            print("  부호 변화 구간 없음 => 교점 없음. 반복 종료.")
            break

        # 첫 번째 부호변화 구간 사용
        idx_sc = sign_change_indices[0]
        v_bef, i1_bef, i2_bef, diff_bef = matched[idx_sc]
        v_aft, i1_aft, i2_aft, diff_aft = matched[idx_sc+1]

        print(f"  [직전] V_DS={v_bef:.4f}, i1={i1_bef:.6g}, i2={i2_bef:.6g}, diff={diff_bef:.6g}")
        print(f"  [직후] V_DS={v_aft:.4f}, i1={i1_aft:.6g}, i2={i2_aft:.6g}, diff={diff_aft:.6g}")

        # (5) 선분 교차점
        inter = find_intersection(v_bef, v_aft, i1_bef, i1_aft,
                                  v_bef, v_aft, i2_bef, i2_aft)
        if inter is None:
            print("  교점이 선분 범위 내에 존재하지 않음 (평행 / 범위 밖 등)")
            break

        x_star, y_star = inter
        print(f"  => 교점: V_DS={x_star:.4f}, I_D={y_star:.6g}")

        # 교점이 0.1×i보다 작은지 검사
        times = 0.1 * i
        print(f"  => 현재까지의 전압강하: {times:.4f}")
        if x_star < times:
            print("  *** 교점 V_DS가 0.1×(횟수)보다 작아졌습니다! ***")
            print("  => 해당 시점에서 반복 중단합니다.")
            found_condition = True

        # --- [추가] 이 iteration의 대칭 곡선 + 교점 정보 기록 ---
        iteration_data.append({
            "iter": i,
            "Vgs2": Vgs2,
            "v2_refl": v2_refl.copy(),  # 혹시 이후에 바뀔까봐 copy()
            "i2_refl": i2_refl.copy(),
            "x_star": x_star,
            "y_star": y_star
        })

        if found_condition:
            break

        # (6) 다음 반복을 위해 Vgs2 -= 0.1
        Vgs2 -= 0.1

    # 최종 결과
    if found_condition:
        print("\n최종 조건을 만족한 시점:")
        print(f"  - 반복 횟수 i={i}")
        print(f"  - Vgs2={Vgs2+0.1:.4f} (다음 감소 직전 값)")
        print(f"  - 교점: V_DS={x_star:.4f}, I_D={y_star:.6g}")
    else:
        print(f"\n주어진 반복({max_iter}회) 동안 교점 V_DS < 0.1×(횟수) 조건이 충족되지 않았습니다.")

    # --------------------------------------
    # (추가) 결과 시각화: 각 iteration의 대칭 곡선 + 교점
    # --------------------------------------
    # 1) 원래 곡선
    plt.figure(figsize=(8,6))
    plt.plot(v1_all, i1_all, 'b-', label=f"Original(Vgs={Vgs1})")

    # 2) iteration마다 대칭 곡선 + 교점
    #    색깔 or 라인스타일은 cycle 되므로, 필요한 경우 수동으로 지정
    for item in iteration_data:
        i_num = item["iter"]
        v2r = item["v2_refl"]
        i2r = item["i2_refl"]
        xs = item["x_star"]
        ys = item["y_star"]
        # 대칭 곡선
        plt.plot(v2r, i2r, label=f"Reflected(i={i_num}, Vgs2={item['Vgs2']:.2f})")
        # 교점
        plt.plot(xs, ys, 'o', ms=6)  # 점 표시 (색은 자동 cycle)

        # iteration 번호를 텍스트로
        plt.text(xs, ys, f"iter={i_num}", fontsize=8, ha='left', va='top')

    plt.xlabel("V_DS (V)")
    plt.ylabel("I_D (A)")
    plt.title(f"Determination of V_FN (Vgs1={Vgs1}, Vgs2={Vgs2+0.1:.2f})")    
    plt.grid(True)
    plt.legend()
    plt.show()
