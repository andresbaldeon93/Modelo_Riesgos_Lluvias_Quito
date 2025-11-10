# ===========================================
#   MODELO DE CLASIFICACIÓN K-MEANS
#   Autor: Andres Baldeon
#   Dataset: Puengasí.csv
# ===========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

url="https://github.com/andresbaldeon93/Modelo_Riesgos_Lluvias_Quito/5c1516717436ed8b60b75487417e48b01f07b16d/Puengasi.csv"
df = pd.read_csv(url, sep='|')

# Revisar estructura básica
print("Columnas:", df.columns.tolist())
#print(df.head())

# 3️⃣ Seleccionar columnas numéricas para clustering
# Ajusta esta lista según tus variables relevantes
#variables = df.head()


# Extraer solo las columnas numéricas
X = df
#print(X)

# Eliminar filas con datos faltantes
X = X.dropna()

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar el número óptimo de clusters con el "método del codo"
inertia = []
K = range(2, 10)  # probamos con 2 a 9 clusters

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

#plt.figure(figsize=(6,4))
#plt.plot(K, inertia, 'bo-')
#plt.xlabel('Número de clusters (k)')
#plt.ylabel('Inercia')
#plt.title('Método del Codo')
#plt.show()


# Entrenar el modelo con el número de clusters elegido
k_optimo = 3  # puedes ajustarlo según el gráfico
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 7️⃣ Evaluar la calidad del clustering con el coeficiente de silueta
score = silhouette_score(X_scaled, df['cluster'])
print(f"Coeficiente de silueta: {score:.3f}")

# 8️⃣ Analizar los resultados
# Promedio de variables por cluster
print(df['cluster'].unique())
df['riesgo']= df['cluster'].map({
    0: 'no riesgo',
    1: 'normal',
    2: 'alto riesgo'
})
print(df['riesgo'].unique())
print(df.head())

#resumen = df.groupby('cluster').mean().round(2)
#print("\nResumen de clusters:")
#print(resumen)

# 9️⃣ Visualización rápida (si las variables tienen sentido espacial o temporal)
sns.scatterplot(x='15_dias', y='riesgo', hue='riesgo', data=df, palette='tab10')
plt.title('Escenarios de lluvias')
plt.show()

# 🔟 Exportar resultados con el cluster asignado
#df.to_csv("Puengasí_clusters.csv", index=False)
#print("\nArchivo exportado: Puengasí_clusters.csv")


