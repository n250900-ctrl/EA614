
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros do sinal
T = 6.0
w0 = 2 * np.pi / T

# Função para definir o sinal original x(t) em um período
def x_t(t):
    conds = [
        (t >= -2) & (t < -1),
        (t >= -1) & (t < 1),
        (t >= 1) & (t < 2)
    ]
    funcs = [
        lambda t: -2 - 2*t,
        lambda t: 1,
        lambda t: -2 + 2*t,
    ]
    return np.piecewise(t, conds, funcs)

# Função para calcular os coeficientes a_k
def get_ak(k):
    if k == 0:
        return 2.0/3.0
    
    term1_sin = np.sin(k * np.pi / 3.0) / (k * np.pi)
    term2_sin = 2 * np.sin(2 * k * np.pi / 3.0) / (k * np.pi)
    term_cos = (6 / (k**2 * np.pi**2)) * (np.cos(2 * k * np.pi / 3.0) - np.cos(k * np.pi / 3.0))
    
    return term1_sin + term2_sin + term_cos

# Função para a série de Fourier truncada
def x_tilde_N(t, N, coeffs):
    sinal_aprox = np.zeros_like(t, dtype=complex)
    for k in range(-N, N + 1):
        sinal_aprox += coeffs[k] * np.exp(1j * k * w0 * t)
    return sinal_aprox.real

# Vetor de tempo para um período
t = np.linspace(-3, 3, 1000)
xt = x_t(t)

# Valores de N
N_values = [1, 10, 20, 50]

# Pré-cálculo dos coeficientes para a maior N
N_max = max(N_values)
coeffs = {k: get_ak(k) for k in range(-N_max, N_max + 1)}

# Geração dos gráficos
for N in N_values:
    xt_approx = x_tilde_N(t, N, coeffs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, xt, label='Sinal Original $x(t)$', linewidth=2)
    plt.plot(t, xt_approx, label=f'Aproximação $\\tilde{{x}}_{{{N}}}(t)$', linestyle='--')
    plt.title(f'Aproximação da Série de Fourier com N = {N}')
    plt.xlabel('Tempo (t)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

#-------------------------------------------------------------------------------------------

print("Potência Média do Erro (P_N):")
for N in N_values:
    xt_approx = x_tilde_N(t, N, coeffs)
    erro = xt - xt_approx
    potencia_erro = np.mean(erro**2)
    print(f'Para N = {N}, P_N = {potencia_erro:.6f}')

#-------------------------------------------------------------------------------------------

N = 50
k_vals = np.arange(-N, N + 1)
omega_vals = k_vals * w0
ak_vals = np.array([coeffs[k] for k in k_vals])

plt.figure(figsize=(12, 6))
plt.stem(omega_vals, np.abs(ak_vals))
plt.title('Módulo dos Coeficientes da Série de Fourier $|a_k|$ para N=50')
plt.xlabel('Frequência Angular $\\omega$ (rad/s)')
plt.ylabel('$|a_k|$')
plt.grid(True)
plt.show()

#--------------------------------------------------------------------------------------------

# Frequência de corte
wc = 10.0

# Vetor de frequência angular
w = np.logspace(-1, 3, 500)

# Resposta em frequência
H_jw = 1 / (1 - 1j * (wc / w))

# Módulo e Fase
H_mag = np.abs(H_jw)
H_phase = np.angle(H_jw)

# Plot do Módulo
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.semilogx(w, H_mag)
plt.title('Módulo da Resposta em Frequência $|H(j\\omega)|$')
plt.xlabel('Frequência $\\omega$ (rad/s)')
plt.ylabel('Ganho')
plt.grid(True, which="both", ls="-")
plt.axvline(wc, color='r', linestyle='--', label='$\\omega_c=10$ rad/s')
plt.legend()

# Plot da Fase
plt.subplot(1, 2, 2)
plt.semilogx(w, H_phase)
plt.title('Fase da Resposta em Frequência $\\angle H(j\\omega)$')
plt.xlabel('Frequência $\\omega$ (rad/s)')
plt.ylabel('Fase (rad)')
plt.grid(True, which="both", ls="-")
plt.axvline(wc, color='r', linestyle='--', label='$\\omega_c=10$ rad/s')
plt.legend()

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------------

N = 50
k_vals = np.arange(-N, N + 1)

# Coeficientes da saída b_k
b_coeffs = {}
for k in k_vals:
    if k == 0:
        # H(j0) = 0 para este filtro passa-altas
        b_coeffs[k] = 0
    else:
        omega_k = k * w0
        H_k = 1 / (1 - 1j * (wc / omega_k))
        b_coeffs[k] = coeffs[k] * H_k

# Construção do sinal de saída y_50(t)
yt_approx = x_tilde_N(t, N, b_coeffs)

# Plot da saída
plt.figure(figsize=(10, 6))
plt.plot(t, yt_approx)
plt.title('Sinal de Saída Aproximado $y_{50}(t)$')
plt.xlabel('Tempo (t)')
plt.ylabel('Amplitude $y(t)$')
plt.grid(True)
plt.show()
