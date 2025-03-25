import numpy as np
import matplotlib.pyplot as plt

# 사용자 입력 (실제 실행 시 콘솔에서 입력)
V_T  = float(input("임계전압 V_T [V]를 입력하세요: "))     # 예: 0.5
V_G2 = float(input("보조 게이트 전압 V_G2 [V]를 입력하세요: "))  # 예: 1.0
V_DS = float(input("드레인-소스 전압 V_DS [V]를 입력하세요: "))  # 예: 2.0

# V_GS1 범위 (0부터 V_DS+V_G2 정도로 잡음)
V_GS1 = np.linspace(0, 15, 500)
I_D = np.zeros_like(V_GS1)
mode_labels = np.empty(V_GS1.shape, dtype=object)

# 각 V_GS1에 대해 모드 결정 및 I_D 계산
for i, v in enumerate(V_GS1):
    if v < V_T:
        # OFF 상태
        mode_labels[i] = "OFF"
        I_D[i] = 0
    else:
        # 먼저 v가 작을 때: T1는 포화 영역 → Mode A 또는 Mode C
        if v < (V_G2 + V_T) / 2:
            # T1 포화 조건: v <= (V_G2+V_T)/2 (이미 만족)
            # T2 조건:
            #   포화: V_DS2 = V_DS - (V_G2 - v) >= V_G2 - V_T  --> v >= 2*V_G2 - V_DS - V_T
            if v >= 2 * V_G2 - V_DS - V_T:
                mode_labels[i] = "A"
                I_D[i] = (v - V_T)**2
            else:
                mode_labels[i] = "C"
                I_D[i] = (v - V_T)**2
        else:
            # v >= (V_G2+V_T)/2 → T1가 선형 영역일 가능성이 있으므로 Mode B 또는 Mode D
            # Mode B: T1 선형, T2 포화
            A_val = (v - V_T) + (V_G2 - V_T)
            disc_B = A_val**2 - 2 * (V_G2 - V_T)**2
            if disc_B < 0:
                # 계산 불가한 경우
                mode_labels[i] = "Undefined"
                I_D[i] = np.nan
            else:
                V_DS1_B = 0.5 * (A_val - np.sqrt(disc_B))
                # T2 포화 조건: V_DS2 = V_DS - V_DS1_B >= V_G2 - V_T  → V_DS1_B <= V_DS - (V_G2 - V_T)
                if V_DS - V_DS1_B >= (V_G2 - V_T):
                    mode_labels[i] = "B"
                    I_D[i] = (V_G2 - V_T - V_DS1_B)**2
                else:
                    # 그렇지 않으면 T2도 선형 → Mode D
                    term_D = 0.5 * (v + V_G2) - V_T
                    rad_D = (V_DS + V_T - 0.5 * (v + V_G2))**2 - V_DS * (V_T - v + 0.5 * V_DS)
                    if rad_D < 0:
                        mode_labels[i] = "Undefined"
                        I_D[i] = np.nan
                    else:
                        V_DS1_D = term_D - np.sqrt(rad_D)
                        # T1 선형: V_DS1_D < v - V_T, T2 선형: V_DS - V_DS1_D < V_G2 - V_T
                        if (V_DS1_D < (v - V_T)) and ((V_DS - V_DS1_D) < (V_G2 - V_T)):
                            mode_labels[i] = "D"
                            I_D[i] = 2 * (v - V_T) * V_DS1_D - V_DS1_D**2
                        else:
                            mode_labels[i] = "Undefined"
                            I_D[i] = np.nan

# 플롯 그리기
plt.figure(figsize=(8, 6))
plt.plot(V_GS1, I_D, 'k-', label='Composite $I_D$')

# 모드별로 점 찍기
unique_modes = np.unique(mode_labels)
for mode in unique_modes:
    idx = np.where(mode_labels == mode)[0]
    if len(idx) > 0:
        if mode == "A":
            plt.scatter(V_GS1[idx], I_D[idx], color='blue', s=15, label='Mode A')
        elif mode == "B":
            plt.scatter(V_GS1[idx], I_D[idx], color='red', s=15, label='Mode B')
        elif mode == "C":
            plt.scatter(V_GS1[idx], I_D[idx], color='green', s=15, label='Mode C')
        elif mode == "D":
            plt.scatter(V_GS1[idx], I_D[idx], color='purple', s=15, label='Mode D')
        elif mode == "OFF":
            plt.scatter(V_GS1[idx], I_D[idx], color='gray', s=15, label='OFF')
        else:
            plt.scatter(V_GS1[idx], I_D[idx], color='orange', s=15, label=mode)

plt.xlabel(r'$V_{GS1}$ [V]')
plt.ylabel(r'$I_D$ [A]')
plt.title(r'MOSFET $I_D$ vs $V_{GS1}$ with Mode Selection' + 
          f'\n($V_T={V_T}$ V, $V_{{G2}}={V_G2}$ V, $V_{{DS}}={V_DS}$ V)')
plt.legend()
plt.grid(True)
plt.show()
