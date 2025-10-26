import numpy as np
import matplotlib.pyplot as plt

def calcula_coeficientes(w, wc, n):
    Tn = np.zeros((w.size,))
    
    idx_pass = np.where(np.abs(w) < wc - 1e-15)
    Tn[idx_pass] = np.cos(n * np.arccos(w[idx_pass] / wc))
    idx_stop = np.where(np.abs(w) >= wc - 1e-15)
    Tn[idx_stop] = np.cosh(n * np.arccosh(w[idx_stop] / wc))
    
    return Tn

def filtro_chebyshev(w, wc, n, epsilon):
    # Calcula os coeficientes Tn primeiro
    Tn = calcula_coeficientes(w, wc, n)
    # Aplica a fórmula da magnitude do filtro
    H_mag = 1.0 / np.sqrt(1 + epsilon**2 * Tn**2)
    return H_mag

def filtro_butterworth(w, wc, n):
    # Aplica a fórmula da magnitude do filtro
    H_mag = 1.0 / np.sqrt(1 + (w / wc)**(2 * n))
    return H_mag

print("Funções auxiliares definidas.")

# -----------------------------------------------------------------
# Item (a): Filtro de Chebyshev vs Ordem (n)
# -----------------------------------------------------------------
print("Gerando gráfico do Item (a)...")
wc_a = 10.0
epsilon_a = 0.2
ordens_n_a = [1, 2, 3, 4, 5]
# Vetor de frequências de 0 a 20 rad/s (1000 pontos)
w_vetor_a = np.linspace(0, 20, 1000) 

# Cria uma nova figura
plt.figure(figsize=(10, 6))
for n_i in ordens_n_a:
    H_c = filtro_chebyshev(w_vetor_a, wc_a, n_i, epsilon_a)
    plt.plot(w_vetor_a, H_c, label=f'n = {n_i}')

plt.title(rf'Filtro de Chebyshev com $\omega_c={wc_a}$ rad/s e $\epsilon={epsilon_a}$')
plt.xlabel(r'Frequência $\omega$ (rad/s)')
plt.ylabel(r'$|H_C(j\omega)|$')
plt.legend()
plt.grid(True)
plt.axvline(wc_a, color='k', linestyle='--', label=r'$\omega_c = 10$')
plt.savefig('chebyshev_ordem_n.png')

# -----------------------------------------------------------------
# Item (b): Filtro de Chebyshev vs Epsilon (epsilon)
# -----------------------------------------------------------------
print("Gerando gráfico do Item (b)...")
wc_b = 10.0
n_b = 3
epsilons_b = [0.1, 0.3, 0.5, 0.7, 0.9]
w_vetor_b = np.linspace(0, 20, 1000)

plt.figure(figsize=(10, 6))
for eps in epsilons_b:
    H_c = filtro_chebyshev(w_vetor_b, wc_b, n_b, eps)
    plt.plot(w_vetor_b, H_c, label=rf'$\epsilon = {eps}$') # 'r' no label

plt.title(rf'Filtro de Chebyshev com $\omega_c={wc_b}$ rad/s e $n={n_b}$')
plt.xlabel(r'Frequência $\omega$ (rad/s)')
plt.ylabel(r'$|H_C(j\omega)|$')
plt.legend()
plt.grid(True)
plt.axvline(wc_b, color='k', linestyle='--', label=r'$\omega_c = 10$')
plt.savefig('chebyshev_epsilon.png')

# -----------------------------------------------------------------
# Item (c): Filtro de Butterworth vs Ordem (n)
# -----------------------------------------------------------------
print("Gerando gráfico do Item (c)...")
wc_c = 10.0
ordens_n_c = [1, 2, 3, 4, 5]
w_vetor_c = np.linspace(0, 20, 1000)

plt.figure(figsize=(10, 6))
for n_i in ordens_n_c:
    H_b = filtro_butterworth(w_vetor_c, wc_c, n_i)
    plt.plot(w_vetor_c, H_b, label=f'n = {n_i}')

plt.title(rf'Filtro de Butterworth com $\omega_c={wc_c}$ rad/s')
plt.xlabel(r'Frequência $\omega$ (rad/s)')
plt.ylabel(r'$|H_B(j\omega)|$')
# Ponto de -3dB (1/sqrt(2))
plt.axhline(1/np.sqrt(2), color='r', linestyle=':', label=r'-3dB (1/$\sqrt{2}$)')
plt.axvline(wc_c, color='k', linestyle='--', label=r'$\omega_c = 10$')
plt.legend()
plt.grid(True)
plt.savefig('butterworth_ordem_n.png')

# -----------------------------------------------------------------
# Item (d): Espectro do Pulso Retangular
# -----------------------------------------------------------------
print("Gerando gráfico do Item (d)...")
wm_d = 7.5
tau_d = 2 * np.pi / wm_d
# Vetor de frequências de 0 a 40 rad/s
w_vetor_d = np.linspace(0, 40, 1000) 

# X(jw) = tau * Sinc_normalizado(w*tau / (2*pi))
# np.sinc(x) já calcula sin(pi*x)/(pi*x) e cuida do caso x=0
X_jw_mag = np.abs(tau_d * np.sinc(w_vetor_d * tau_d / (2 * np.pi)))

plt.figure(figsize=(10, 6))
plt.plot(w_vetor_d, X_jw_mag)
plt.title(rf'Magnitude $|X(j\omega)|$ do Pulso Retangular ($\omega_m=7.5$)')
plt.xlabel(r'Frequência $\omega$ (rad/s)')
plt.ylabel(r'$|X(j\omega)|$')
# Marca os zeros calculados
zeros = [wm_d * k for k in range(1, 6)]
plt.scatter(zeros, np.zeros_like(zeros), color='red', zorder=5, 
            label=r'Zeros ($k \cdot \omega_m$)')
plt.legend()
plt.grid(True)
plt.savefig('pulso_retangular_espectro.png')

# -----------------------------------------------------------------
# Item (e): Filtragem e Comparação
# -----------------------------------------------------------------
print("Gerando gráficos do Item (e)...")
w_vetor_e = np.linspace(0, 40, 2000) # Mais pontos para melhor definição
wc_e = 10.0

# 1. Sinal de entrada X(jw)
wm_e = 7.5
tau_e = 2 * np.pi / wm_e
X_jw_mag_e = np.abs(tau_e * np.sinc(w_vetor_e * tau_e / (2 * np.pi)))

# 2. Definir os tres filtros
# Filtro Ideal
H_ideal = np.zeros_like(w_vetor_e)
H_ideal[w_vetor_e <= wc_e] = 1.0

# Filtro Chebyshev (n=4, epsilon=0.6)
H_cheby = filtro_chebyshev(w_vetor_e, wc_e, n=4, epsilon=0.6)

# Filtro Butterworth (n=2)
H_butter = filtro_butterworth(w_vetor_e, wc_e, n=2)

# 3. Calcular as saidas Y(jw) = H(jw) * X(jw)
Y_ideal = H_ideal * X_jw_mag_e
Y_cheby = H_cheby * X_jw_mag_e
Y_butter = H_butter * X_jw_mag_e

# --- Plot 1: Resposta em Frequência dos Filtros ---
plt.figure(figsize=(12, 6))
plt.plot(w_vetor_e, H_ideal, 'r--', label=r'Ideal ($\omega_c=10$)')
plt.plot(w_vetor_e, H_cheby, 'g', label=r'Chebyshev ($n=4, \epsilon=0.6$)')
plt.plot(w_vetor_e, H_butter, 'b:', label=r'Butterworth ($n=2$)')
plt.title('Respostas em Frequência dos Três Filtros')
plt.xlabel(r'Frequência $\omega$ (rad/s)')
plt.ylabel(r'$|H(j\omega)|$')
plt.legend()
plt.grid(True)
plt.ylim(-0.05, 1.1) # Garante uma boa visualização
plt.savefig('filtros_comparacao.png')

# --- Plot 2: Espectro das Saídas ---
plt.figure(figsize=(12, 6))
plt.plot(w_vetor_e, Y_ideal, 'r--', label=r'Saída $Y_{ideal}$')
plt.plot(w_vetor_e, Y_cheby, 'g', label=r'Saída $Y_{Chebyshev}$')
plt.plot(w_vetor_e, Y_butter, 'b:', label=r'Saída $Y_{Butterworth}$')
# Plota a entrada original com transparência para referência
plt.plot(w_vetor_e, X_jw_mag_e, 'k', alpha=0.3, label=r'Entrada $|X(j\omega)|$') 
plt.title('Espectro de Magnitude das Saídas dos Filtros')
plt.xlabel(r'Frequência $\omega$ (rad/s)')
plt.ylabel(r'$|Y(j\omega)|$')
plt.legend()
plt.grid(True)
plt.savefig('saidas_comparacao.png')

print("\nTodos os 6 gráficos foram gerados e salvos com sucesso na pasta do script.")
print("Exibindo gráficos...")
# Exibe todas as figuras criadas
plt.show()
