import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

# Cargar el modelo
modelo_rf = joblib.load('modelos/trained_models/random_forest_model.pkl')

# Obtener la importancia de las características
importancias = modelo_rf.feature_importances_
nombres_caracteristicas = modelo_rf.feature_names_in_

# Crear un DataFrame para facilitar la visualización
df_importancias = pd.DataFrame({'caracteristica': nombres_caracteristicas, 'importancia': importancias})
df_importancias = df_importancias.sort_values('importancia', ascending=False)

# Crear la gráfica de importancia de características
plt.figure(figsize=(12, 6))
plt.bar(df_importancias['caracteristica'], df_importancias['importancia'])
plt.title('Importancia de Características - Random Forest')
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('importancia_caracteristicas.png')
plt.show()

# Visualizar un árbol individual del bosque
plt.figure(figsize=(20,10))
plot_tree(modelo_rf.estimators_[0], feature_names=nombres_caracteristicas, filled=True, rounded=True, max_depth=3)
plt.savefig('arbol_ejemplo.png')
plt.show()

# Imprimir algunas estadísticas del modelo
print(f"Número de árboles en el bosque: {modelo_rf.n_estimators}")
print(f"Profundidad máxima de los árboles: {modelo_rf.max_depth}")
print(f"Número de características: {modelo_rf.n_features_in_}")
print("\nImportancia de características:")
for caracteristica, importancia in zip(nombres_caracteristicas, importancias):
    print(f"{caracteristica}: {importancia:.4f}")