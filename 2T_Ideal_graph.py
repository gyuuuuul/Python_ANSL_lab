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
# 메인 설정 (전역 변수로 이동)
# 2) 메인 설정
##########################################
 
V_T = 0.5
v_ds_list = [1, 2, 3, 4]
N = 400  # 해상도 증가
 
v_gs1_vals = np.linspace(0, 5, N)
v_g2_vals = np.linspace(0, 5, N)

plt.rcParams['font.family'] = 'Arial'  # Arial 폰트 전체 적용


# ▶▶▶ 추가할 부분 ▶▶▶
mode_labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
# ◀◀◀◀◀◀◀◀◀◀◀◀◀◀

# X축(V_GS1), Y축(V_G2)의 범위 (전역 설정)
v_gs1_vals = np.linspace(0, 5, N)
v_g2_vals = np.linspace(0, 5, N)
X, Y = np.meshgrid(v_gs1_vals, v_g2_vals, indexing='xy')

fig, axes = plt.subplots(2, 2, figsize=(10,8))
axes = axes.flatten()

for idx, vds_fixed in enumerate(v_ds_list):
    ax = axes[idx]
    ax.set_title(f'V$_{{DS}}$ = {vds_fixed} V', fontsize=12)
    ax.set_xlabel('V$_{GS1}$', fontsize=10)
    ax.set_ylabel('V$_{G2}$', fontsize=10)
    
    # 축 두께 설정
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 모드 맵 계산
    mode_map = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            mode_map[i,j] = check_mode(vds_fixed, Y[i,j], X[i,j], V_T)
    
    # 컨투어 그리기 (내부 경계선 0.5, 바깥 경계선 2)
    ax.contour(X, Y, mode_map, 
               levels=[0.5, 1.5, 2.5, 3.5, 4.5], 
               colors='k', linewidths=0.5)
    ax.contour(X, Y, mode_map, 
               levels=[0.5, 4.5], 
               colors='k', linewidths=2)
    
    # 영역 라벨 추가
    for m in [1, 2, 3, 4]:
        mask = (mode_map == m)
        if np.any(mask):
            x_mean = X[mask].mean()
            y_mean = Y[mask].mean()
            ax.text(x_mean, y_mean, mode_labels[m],
                    fontsize=14, ha='center', va='center',
                    fontweight='bold', color='k')

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

plt.tight_layout()
plt.show()
