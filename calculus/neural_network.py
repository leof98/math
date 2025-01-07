import math
import csv
import matplotlib.pyplot as plt
import random

# https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
path = "./heart_failure.csv"

dataset_x = []
dataset_y = []

with open(path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader) # Está pulando o cabeçalho

    for row in reader:
        try:
            age = float(row[0])
            death = int(row[-1])
            dataset_x.append(age)
            dataset_y.append(death)
        except ValueError:
            # Pula as linhas que estão com os dados inválidos
            continue


def normalize(data):
    mean = sum(data) / len(data)
    variancia = sum((xi - mean) ** 2 for xi in data) / len(data)
    std_dev = variancia ** 0.5

    normalized_data = [(xi - mean) / std_dev for xi in data]
    return normalized_data, mean, std_dev


dataset_x, mean_x, std_dev_x = normalize(dataset_x)


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def df_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cal_f1score(dataset_x, dataset_y, a, b):
    n = len(dataset_x)

    verdadeiro_positivo = 0
    falso_positivo = 0
    falso_negativo = 0

    for i in range(n):
        x = dataset_x[i]
        y_dt = dataset_y[i]

        z = a*x + b

        if sigmoid(z) >= 0.5:
            y_prd = 1
        else:
            y_prd = 0
        
        if y_prd == 1 and y_dt == 1:
            verdadeiro_positivo += 1
        elif y_prd == 1 and y_dt == 0:
            falso_positivo += 1
        elif y_prd == 0 and y_dt == 1:
            falso_negativo += 1
    
    # precisão
    if (verdadeiro_positivo + falso_positivo) > 0:
        precisao = verdadeiro_positivo / (verdadeiro_positivo + falso_positivo)
    else:
        precisao = 0
    
    # recall
    if (verdadeiro_positivo + falso_negativo) > 0:
        recall = verdadeiro_positivo / (verdadeiro_positivo + falso_negativo)
    else:
        recall = 0
    
    if precisao + recall > 0:
        return 2 * (precisao * recall) / (precisao + recall)
    else:
        return 0


def derivadas(dataset_x, dataset_y, a, b, c, d, k, l):
    n = len(dataset_x)
    df_a, df_b, df_c, df_d, df_k, df_l = 0, 0, 0, 0, 0, 0

    for i in range(n):
        x = dataset_x[i]
        y = dataset_y[i]

        u = sigmoid(a*x + b)
        w = sigmoid(c*u + d)
        z = sigmoid(k*w + l)
                    
        erro = y - z

        # gradientes
        dz = -2 * erro * df_sigmoid(k*w + l)
        dw = dz * k * df_sigmoid(c*u + d)
        du = dw * c * df_sigmoid(a*x + b)

        # parâmetros
        df_a += du * x
        df_b += du
        df_c += dw * u
        df_d += dw
        df_k += dz * w
        df_l += dz

    return df_a, df_b, df_c, df_d, df_k, df_l


def distancia(a1, b1, c1, d1, k1, l1, a0, b0, c0, d0, k0, l0):
    return ((a1 - a0)**2 + (b1 - b0)**2 + (c1 - c0)**2 + (d1 - d0)**2 + (k1 - k0)**2 + (l1 - l0)**2) ** 0.5


def grad_des(a, b, c, d, k, l, tolerancia, learning_rate):
    a0 = a
    b0 = b
    c0 = c
    d0 = d
    k0 = k
    l0 = l

    a1, b1, c1, d1, k1, l1 = [0.1] * 6

    i = 0
    loading_state = 0

    while True:
        df = derivadas(dataset_x, dataset_y, a0, b0, c0, d0, k0, l0)

        a1 = a0 - learning_rate * df[0]
        b1 = d0 - learning_rate * df[1]
        c1 = c0 - learning_rate * df[2]
        d1 = d0 - learning_rate * df[3]
        k1 = k0 - learning_rate * df[4]
        l1 = l0 - learning_rate * df[5]

        dist = distancia(a1, b1, c1, d1, k1, l1, a0, b0, c0, d0, k0, l0)

        i += 1

        # Layout de carregamento
        if i % 1000 == 0:
            loading_state = (loading_state + 1) % 4
            if loading_state == 0:
                print("Carregando   ", end="\r")  # Limpa a linha
            elif loading_state == 1:
                print("Carregando.  ", end="\r")
            elif loading_state == 2:
                print("Carregando.. ", end="\r")
            elif loading_state == 3:
                print("Carregando...", end="\r")

        
        if dist > tolerancia:
            a0 = a1
            b0 = b1
            c0 = c1
            d0 = d1
            k0 = k1
            l0 = l1
        
        else:
            break

    return i, a0, b0, c0, d0, k0, l0




a1, b1, c1, d1, k1, l1 = [random.uniform(-1, 1) for _ in range(6)]

tolerancia = 10**(-6)
learning_rate = 0.01

passos, a_final, b_final, c_final, d_final, k_final, l_final = grad_des(a1, b1, c1, d1, k1, l1, tolerancia, learning_rate)



pred = []

for x in dataset_x:
    u = sigmoid(a_final * x + b_final)
    w = sigmoid(c_final * u + d_final)
    z = sigmoid(k_final * w + l_final)
    # Limiarização para classificação binária
    pred.append(1 if z >= 0.5 else 0)

acertos = sum([1 if p == y else 0 for p, y in zip(pred, dataset_y)])
total = len(dataset_y)
acuracia = acertos / total
f1score_final = cal_f1score(dataset_x, dataset_y, a_final, b_final)


print('')
print(f'Total de interações: {passos}')
print(f'Acurácia: {acuracia:.3%}')
print(f'F1-Score: {f1score_final:.2f}')


# Gráfico
dados_ordenados = sorted(zip(dataset_x, dataset_y, pred), key=lambda x: x[0])

# Separe os dados ordenados novamente
dataset_x_ordenado, dataset_y_ordenado, pred_ordenado = zip(*dados_ordenados)

plt.plot(dataset_x, dataset_y, 'bo', label='Real')
plt.plot(dataset_x_ordenado, pred_ordenado, 'r-', label='Predição')

plt.xlabel('Idade normalizada')
plt.ylabel('Morte')
plt.title('Predição usando rede neural')
plt.legend()
plt.grid(True)
plt.show()
