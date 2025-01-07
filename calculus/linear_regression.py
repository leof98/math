import matplotlib.pyplot as plt

dataset_x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
dataset_y = [0, 1, 2, 3, 4, 5, 6, 7]

coef = [1, 1, 1, 1, 1, 1, 1, 1]


# Gradiente
def gd(dataset_x, dataset_y, coef):
    n = len(coef)
    gradientes = [0] * n

    for i in range(len(dataset_x)):

        x = dataset_x[i]
        y = dataset_y[i]

        for k in range(n):
            soma = 0

            for j in range(n):
                soma += coef[j] * (x ** (n - j - 1))
            
            gradientes[k] += 2 * (y - soma) * -(x ** (7 - k))
    
    return gradientes


# Função para calcular a distância
def distancia(coef1, coef2):
    return sum((coef1[i] - coef2[i]) ** 2 for i in range(len(coef1))) ** 0.5


# Gradiente Descendente: (Xn+1 = Xn - Lr * gd)
def gds(dataset_x, dataset_y, coef, lr, tol):
    n = len(coef)
    Xn = [9999] * n  # Chute Inicial
    Xn1 = coef

    c = 1
    loading_state = 0
    dist = distancia(Xn, Xn1)

    while dist > tol:

        grad = gd(dataset_x, dataset_y, coef)
        Xn = Xn1 + []  # Substitui os valores de Xn para Xn+1

        for i in range(n):
            Xn1[i] = Xn[i] - lr * grad[i] / n

        dist = distancia(Xn, Xn1)
        c += 1

        # Layout de carregamento
        if c % 10000 == 0:
            loading_state = (loading_state + 1) % 4
            if loading_state == 0:
                print("Carregando   ", end="\r")  # Limpa a linha
            elif loading_state == 1:
                print("Carregando.  ", end="\r")
            elif loading_state == 2:
                print("Carregando.. ", end="\r")
            elif loading_state == 3:
                print("Carregando...", end="\r")
    
    return (c, Xn1)


lr = 1e-4
tol = 5e-6

g = gds(dataset_x, dataset_y, coef, lr, tol)

# Montando o gráfico
x_vals = [i * 0.01 for i in range(71)] 
y_vals = [sum(g[1][i] * (x ** (7 - i)) for i in range(len(g[1]))) for x in x_vals]

# Plotar
plt.figure(figsize=(10, 6))
plt.scatter(dataset_x, dataset_y, color='red', label='Dados originais')
plt.plot(x_vals, y_vals, color='blue', label='Ajuste (Gradiente Descendente)')
plt.title('Ajuste Polinomial usando Gradiente Descendente', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid()
plt.show()
