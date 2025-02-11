import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelEncoder

# Carrega Dataset
dataset = pd.read_csv('DatasetPCA/PCAeco.csv')

# Muda de string para float
for column in dataset.columns[1:]:
    dataset[column] = dataset[column].str.replace(',', '').astype(float)

# Cria um dataframe
data = pd.DataFrame(data=dataset)
data['target'] = dataset.País[dataset.index]

# Separa os números dos rótulos
y = data['target'].values
X = data.iloc[:, 1:5].values

# Normalizar os dados
X = (X - np.mean(X)) / np.std(X)

# Calcular e imprimir a matriz de covariância
cov_matrix = np.cov(X.T)
print("Matriz de Covariância:")
print(cov_matrix)

# Calcular e imprimir os autovalores e autovetores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nAutovalores:")
print(eigenvalues)
print("\nAutovetores:")
print(eigenvectors)

# Ordenar os autovalores e autovetores em ordem decrescente
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) 
               for i in range(len(eigenvalues))
              ]
eigen_pairs.sort(key=lambda x: x[0], reverse=True)

# Selecionar os dois maiores autovalores e autovetores
top_eigenvalues = eigen_pairs[0][0], eigen_pairs[1][0]
top_eigenvectors = eigen_pairs[0][1], eigen_pairs[1][1]

print("\nDois maiores autovalores:")
print(top_eigenvalues)
print("\nDois maiores autovetores:")
print(top_eigenvectors)

# Aplica o PCA com 2 componentes principais
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Converte os rótulos para valores numéricos
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Obtem cores únicas para cada classe
num_classes = len(np.unique(y))
colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

# Cria um mapeamento de classe para cor
class_color_map = {class_label: color for class_label, color in zip(np.unique(y), colors)}

# Visualiza os resultados
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='viridis')

# Adiciona legendas para as classes
legend_handles = [Patch(color=class_color_map[label], label=label) for label in np.unique(y)]

plt.legend(handles=legend_handles, title="Classes")

plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('PCA')
plt.show()
