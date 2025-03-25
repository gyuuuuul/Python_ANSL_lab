import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

#########################################
# 1) 모드 조건 & V_DS1 식 정의
#########################################

V_T = 0.5  # 문제에서 V_TH=0.5

def modeA_condition(vds, vgs1, vg2, vt=V_T):
    """ Mode A 조건 """
    return (
        (vgs1 < (vg2 + vt)*0.5) and
        (vgs1 > vt) and
        ((vds + vt) > vg2)
    )

def modeA_vds1(vds, vgs1, vg2, vt=V_T):
    """ Mode A: V_DS1 = V_G2 - V_GS1 """
    return vg2 - vgs1

def modeB_condition(vds, vgs1, vg2, vt=V_T):
    """ Mode B 조건 """
    return (
        (vgs1 > (vg2 + vt)*0.5) and
        (vg2 > vt) and
        (vds > (vg2 - vt))
    )

def modeB_vds1(vds, vgs1, vg2, vt=V_T):
    """
      V_DS1 = 0.5 * [ (v_gs1 - vt) + (v_g2 - vt)
                      - sqrt( [ (v_gs1-vt)+(v_g2-vt) ]^2 - 2*(v_g2 - vt)^2 ) ]
    """
    A = (vgs1 - vt) + (vg2 - vt)
    B = (vg2 - vt)
    inside = A**2 - 2*(B**2)
    if inside < 0:
        return np.nan
    return 0.5 * ( A - np.sqrt(inside) )

def modeC_condition(vds, vgs1, vg2, vt=V_T):
    """ Mode C 조건 """
    exprC = 2*(vg2 - vt)*(vgs1 - (vds + vt)) + vds**2
    return (
        (exprC < 0) and
        (vg2 > vgs1) and
        (vgs1 > vt) and
        (vg2 > vds + vt)
    )

def modeC_vds1(vds, vgs1, vg2, vt=V_T):
    """ V_DS1 = (v_g2 - vt) - sqrt( (v_ds + vt - v_g2)^2 + (v_gs1 - vt)^2 ) """
    inside = (vds + vt - vg2)**2 + (vgs1 - vt)**2
    return (vg2 - vt) - np.sqrt(inside)

def modeD_condition(vds, vgs1, vg2, vt=V_T):
    """ Mode D 조건 """
    exprD = 2*(vg2 - vt)*(vgs1 - (vds + vt)) + vds**2
    return (
        (exprD > 0) and
        (vg2 > vds + vt)
    )

def modeD_vds1(vds, vgs1, vg2, vt=V_T):
    """
      V_DS1 = 0.5 * [ (v_gs1+v_g2) - 2*vt
                      - sqrt( [v_ds+vt-0.5(v_gs1+v_g2)]^2
                               - v_ds*(vt - v_gs1 + 0.5*v_ds ) ) ]
    """
    half_sum = 0.5*(vgs1 + vg2)
    inside = (vds + vt - half_sum)**2 - vds*(vt - vgs1 + 0.5*vds)
    if inside < 0:
        return np.nan
    return 0.5*((vgs1+vg2) - 2*vt - np.sqrt(inside))

def get_mode_and_vds1(vds, vgs1, vg2):
    """
    (vds, vgs1, vg2)에 대해 모드 판정.
    - 여러 모드를 동시에 만족한다면, 아래 순서(A→B→C→D)로 먼저 맞는 모드 반환.
    - 반환: (mode번호, v_ds1)
    """
    # A
    if modeA_condition(vds, vgs1, vg2):
        return 1, modeA_vds1(vds, vgs1, vg2)
    # B
    if modeB_condition(vds, vgs1, vg2):
        return 2, modeB_vds1(vds, vgs1, vg2)
    # C
    if modeC_condition(vds, vgs1, vg2):
        return 3, modeC_vds1(vds, vgs1, vg2)
    # D
    if modeD_condition(vds, vgs1, vg2):
        return 4, modeD_vds1(vds, vgs1, vg2)

    # 어느 모드도 안 맞음
    return 0, np.nan


#########################################
# 2) (V_GS1, V_G2) 격자에서 모드 판정 후 V_DS1 계산
#########################################

def compute_surface_for_vds(vds_fixed, N=150):
    """
    vds_fixed (float): V_DS 값을 고정
    N: 격자 해상도
    return: (X, Y, Z, mode_map)
      - X, Y shape=(N,N)  -> meshgrid(V_GS1, V_G2)
      - Z  shape=(N,N)    -> 각 점에서의 V_DS1 (NaN이면 미정)
      - mode_map shape=(N,N) -> A=1, B=2, C=3, D=4, 그 외=0
    """
    vgs1_vals = np.linspace(0, 5, N)  # x축
    vg2_vals  = np.linspace(0, 5, N)  # y축
    X, Y = np.meshgrid(vgs1_vals, vg2_vals, indexing='xy')

    Z = np.full((N,N), np.nan)
    mode_map = np.zeros((N,N), dtype=int)

    for i in range(N):
        for j in range(N):
            v_gs1 = X[i,j]
            v_g2  = Y[i,j]
            m, z_val = get_mode_and_vds1(vds_fixed, v_gs1, v_g2)
            mode_map[i,j] = m
            Z[i,j] = z_val

    return X, Y, Z, mode_map

#########################################
# 3) 모드별 색 + 경계선에 점선 (z=0까지)
#########################################

def draw_vertical_dashed_lines(ax, X, Y, Z, mode_map):
    """
    mode_map!=0 인 유효영역의 "경계점"에 대해,
    그 점의 (x,y,z)에서 (x,y,0)까지 점선으로 그린다.

    여기서는 간단히:
     - 만약 (i,j)가 모드!=0인데
     - 상하좌우 이웃 중 하나라도 모드=0(또는 범위 밖)이면
       => 경계점으로 보고, 수직 점선 표시

    ※ 모드가 바뀌는 내부 경계도 보고 싶다면,
       "이웃 모드!=mode_map[i,j]"로 체크하면 된다.
    """
    N = X.shape[0]
    for i in range(N):
        for j in range(N):
            if mode_map[i,j] != 0 and not np.isnan(Z[i,j]):
                neighbors_mode = []
                for di,dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ni, nj = i+di, j+dj
                    # 범위 밖이면 경계 취급
                    if not (0 <= ni < N and 0 <= nj < N):
                        neighbors_mode.append(0)
                    else:
                        neighbors_mode.append(mode_map[ni,nj])
                # 이웃 중 하나라도 0이면 경계
                if any(m==0 for m in neighbors_mode):
                    x = X[i,j]
                    y = Y[i,j]
                    z = Z[i,j]
                    ax.plot([x, x], [y, y], [0, z],
                            color='k', linestyle='--', linewidth=0.5)

#########################################
# 4) 실제로 4개 (V_DS=1,2,3,4) 서브플롯 그리기
#########################################

import matplotlib.pyplot as plt

# 모드→색 (0: 흰색, 1:A=빨강, 2:B=초록, 3:C=파랑, 4:D=마젠타)
cmap_list = ['white','red','green','blue','magenta']
cmap = mcolors.ListedColormap(cmap_list)
bounds = [0,1,2,3,4,5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

vds_list = [1,2,3,4]
fig = plt.figure(figsize=(12,10))

for idx, vds_fixed in enumerate(vds_list):
    X, Y, Z, mode_map = compute_surface_for_vds(vds_fixed, N=500)

    ax = fig.add_subplot(2, 2, idx+1, projection='3d')
    ax.set_title(f"V_DS = {vds_fixed} V (V_TH=0.5V)")
    ax.set_xlabel("V_GS1")
    ax.set_ylabel("V_G2")
    ax.set_zlabel("V_DS1")

    # facecolors 생성
    #  (mode_map[i,j]를 컬러로 매핑 -> (N*N,4) -> reshape (N,N,4))
    mode_flat = mode_map.flatten()
    color_rgba = cmap(norm(mode_flat))  # shape=(N*N,4)
    facecolors_3d = color_rgba.reshape((X.shape[0], X.shape[1], 4))

    # 3D 표면 표시
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors_3d,
        linewidth=0, rstride=1, cstride=1,
        antialiased=False
    )
    ax.set_zlim([-1, 5])  # 보기 좋게 적당히 범위 설정

    # 경계에 점선 표시
    draw_vertical_dashed_lines(ax, X, Y, Z, mode_map)

    # 모드 범례를 붙이고 싶으면 colorbar 활용
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])  # 값은 dummy
    cb = plt.colorbar(mappable, ax=ax, fraction=0.05, pad=0.07)
    cb.set_ticks([0.5,1.5,2.5,3.5,4.5])
    cb.set_ticklabels(['None','A','B','C','D'])

plt.tight_layout()
plt.show()
