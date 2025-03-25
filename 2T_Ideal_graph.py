import numpy as np
import matplotlib.pyplot as plt

##########################################
# 1) 모드 판정 함수
##########################################
def check_mode(v_ds, v_g2, v_gs1, v_t=0.5):
    """
    (v_ds, v_g2, v_gs1)가 주어졌을 때
    주어진 Mode A/B/C/D 조건을 판정하여
    A→1, B→2, C→3, D→4 (그 외→0)을 리턴.
    """
    # -- Mode A --
    #   1) v_gs1 < (v_g2 + v_t)/2
    #   2) v_gs1 > v_t
    #   3) v_ds + v_t > v_g2
    if (v_gs1 < (v_g2 + v_t)/2) and (v_gs1 > v_t) and ((v_ds + v_t) > v_g2):
        return 1  # Mode A

    # -- Mode B --
    #   1) v_gs1 > (v_g2 + v_t)/2
    #   2) v_g2 > v_t
    #   3) v_ds > v_g2 - v_t
    if (v_gs1 > (v_g2 + v_t)/2) and (v_g2 > v_t) and (v_ds > (v_g2 - v_t)):
        return 2  # Mode B

    # -- Mode C --
    #   1) 2(v_g2 - v_t)[v_gs1 - (v_ds + v_t)] + v_ds^2 < 0
    #   2) v_g2 > v_gs1
    #   3) v_gs1 > v_t
    #   4) v_g2 > v_t + v_ds
    exprC = 2*(v_g2 - v_t)*(v_gs1 - (v_ds + v_t)) + v_ds**2
    if (exprC < 0) and (v_g2 > v_gs1) and (v_gs1 > v_t) and (v_g2 > (v_t + v_ds)):
        return 3  # Mode C

    # -- Mode D --
    #   1) 2(v_g2 - v_t)[v_gs1 - (v_ds + v_t)] + v_ds^2 > 0
    #   2) v_g2 > v_ds + v_t
    exprD = 2*(v_g2 - v_t)*(v_gs1 - (v_ds + v_t)) + v_ds**2
    if (exprD > 0) and (v_g2 > (v_ds + v_t)):
        return 4  # Mode D

    return 0  # 어느 모드에도 해당 안 됨

##########################################
# 2) 메인 설정
##########################################
V_T = 0.5
v_ds_list = [1, 2, 3, 4]  # 고정할 V_DS 값들
N = 200                   # 2D 격자 해상도

# X축(V_GS1), Y축(V_G2)의 범위
v_gs1_vals = np.linspace(0, 5, N)
v_g2_vals  = np.linspace(0, 5, N)

# 2D 격자: X[i,j] = V_GS1, Y[i,j] = V_G2
X, Y = np.meshgrid(v_gs1_vals, v_g2_vals, indexing='xy')

# 모드→문자 매핑
mode_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

##########################################
# 3) 4개 서브플롯(각각 V_DS = 1,2,3,4)
##########################################
fig, axes = plt.subplots(2, 2, figsize=(10,8))
axes = axes.flatten()

for idx, vds_fixed in enumerate(v_ds_list):
    ax = axes[idx]
    ax.set_title(f'V_DS = {vds_fixed} V')
    ax.set_xlabel('V_GS1')
    ax.set_ylabel('V_G2')

    # 각 (x,y)마다 모드 판정 결과를 담을 2D 배열
    mode_map = np.zeros((N, N), dtype=int)

    # 그리드 각 지점에서 체크
    for i in range(N):
        for j in range(N):
            v_gs1 = X[i,j]
            v_g2  = Y[i,j]
            mode_map[i,j] = check_mode(vds_fixed, v_g2, v_gs1, V_T)

    # --------------------------------------------------
    # (1) 경계선(contour)만 그리기
    # 모드가 정수 1,2,3,4,0 이므로,
    #   levels = [0.5, 1.5, 2.5, 3.5, 4.5] 근방을 contour로 잡으면
    #   모드 1,2,3,4 경계선이 그려진다.
    # --------------------------------------------------
    CS = ax.contour(
        X, Y, mode_map,
        levels=[0.5, 1.5, 2.5, 3.5, 4.5],
        colors='k'   # 검정색 경계선
    )

    # --------------------------------------------------
    # (2) 각 모드 영역에 문자(A,B,C,D) 찍기
    #     여기서는 한 모드가 여러 개의 불연속 영역이면
    #     중앙 한 곳에만 표시된다는 점 유의
    # --------------------------------------------------
    for m in [1,2,3,4]:
        mask = (mode_map == m)
        if np.any(mask):
            # 이 영역의 (x,y) 평균 위치에 라벨
            x_mean = np.mean(X[mask])
            y_mean = np.mean(Y[mask])
            ax.text(x_mean, y_mean, mode_labels[m],
                    color='k', fontsize=14,
                    ha='center', va='center',
                    fontweight='bold')

    ax.set_xlim(v_gs1_vals[0], v_gs1_vals[-1])
    ax.set_ylim(v_g2_vals[0],  v_g2_vals[-1])

plt.tight_layout()
plt.show()
