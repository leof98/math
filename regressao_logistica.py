import math
import csv
import matplotlib.pyplot as plt

path = "./src/heart_failure.csv"

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


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def grad_a(a,b,x,y):
    z = a*x + b
    e = sigmoid(z)
    return (e - y) * x

def grad_b(a,b,x,y):
    z = a*x + b
    e = sigmoid(z)
    return (e - y)


def distancia(a0,b0,a1,b1):
    return ((a1-a0)**2 + (b1-b0)**2) ** 0.5


def cal_acuracia(dataset_x,dataset_y,a,b):
    n = len(dataset_x)
    corretos = 0

    for i in range(n):
        x = dataset_x[i]
        y_dt = dataset_y[i]
        
        z = a*x + b

        if sigmoid(z) >= 0.5:
            y_prd = 1
        else:
            y_prd = 0
        
        if y_prd == y_dt:
            corretos += 1
        
    return corretos / n


def cal_f1score(dataset_x,dataset_y,a,b):
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


def gradiente(dataset_x,dataset_y,a0,b0,lr,tol):
    n = len(dataset_x)
    a, b = a0, b0

    cost_copy = []
    i = 0

    while True:
        grad_a_total = 0
        grad_b_total = 0
        cost = 0

        for i in range(n):
            x = dataset_x[i]
            y = dataset_y[i]

            z = a*x + b

            prd = sigmoid(z)
            error = y - prd

            grad_a_total += error *  x
            grad_b_total += error

            if prd == 0:
                prd = 1e-15
            elif prd == 1:
                prd = 1 - 1e-15
            
            cost += -y * math.exp(prd) - (1 - y) * math.log(1 - prd)

        # Média da função de custo
        cost /= n
        cost_copy.append(cost)

        a_new = a + lr * grad_a_total / n
        b_new = b + lr * grad_b_total / n

        d = distancia(a,b,a_new,b_new)

        if d < tol:
            print(f"Passo atingido: {i + 1}")
            break

        a, b = a_new, b_new
        i += 1

    return a, b, i, cost_copy


# Normalização dos dados
def normalize(data):
    mean = sum(data) / len(data)
    variancia = sum((xi - mean) ** 2 for xi in data) / len(data)
    std_dev = variancia ** 0.5

    return [(xi - mean) / std_dev for xi in data]


# Normalizando o xi (age)
normalized_x = normalize(dataset_x)


a_inicial = 0
b_inicial = 0
learning_rate = 1e-2
tolerancia = 1e-9


print("Carregando...")


a_final, b_final, passos, cost_copy = gradiente(normalized_x, dataset_y, a_inicial, b_inicial, learning_rate, tolerancia)


cost_final = cost_copy[-1]
acuracia_final = cal_acuracia(normalized_x, dataset_y, a_final, b_final)
f1score_final = cal_f1score(normalized_x, dataset_y, a_final, b_final)


print(f"\nTotal de interações: {passos}")
print(f"Custo final: {cost_final:.6f}")
print(f"Acurácia final: {acuracia_final:.3%}")
print(f"F1-Score final: {f1score_final:.2f}")


x_plot = [normalized_x[i] for i in sorted(range(len(normalized_x)), key=lambda i: normalized_x[i])]
y_plot = [1 if sigmoid(a_final * xi + b_final) >= 0.5 else 0 for xi in x_plot]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(normalized_x, dataset_y, alpha=0.6, label='Dados reais', color='blue')
plt.plot(x_plot, y_plot, color='red', label='Curva Sigmoide Ajustada (classificação binária)')
plt.title("Regressão Logística - Idade Normalizada vs. Morte")
plt.xlabel("Idade Normalizada")
plt.ylabel("Classificação de Morte (0 ou 1)")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cost_copy, color='green')
plt.title("Histórico da Função de Custo (Log-loss)")
plt.xlabel("Iterações")
plt.ylabel("Custo")
plt.grid(True)

plt.tight_layout()
plt.show()
