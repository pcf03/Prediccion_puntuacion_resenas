import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 0

# --- FASE 2: ALE ---

# 1. TOMAMOS LA MATRIZ DE PAULA (Ya limpia y reducida por UMAP)
datos = np.load("resultados/datos_procesados.npz")
X_train = datos['X_train']
X_test = datos['X_test']
y_train = datos['y_train']
y_test = datos['y_test']

# 2. CONFIGURACIÓN DE MODELOS Y VALIDACIÓN
validacion = KFold(n_splits=5, shuffle=True, random_state=SEED)

configuraciones = {
    "Dummy (mediana)": DummyRegressor(strategy="median"),
    "Ridge (alpha=1.0)": Pipeline([
        ("escala", StandardScaler()),
        ("modelo", Ridge(alpha=1.0, solver="lsqr"))
    ]),
    "SVR (C=1.0)": Pipeline([
        ("escala", StandardScaler()),
        ("modelo", SVR(kernel="rbf", C=1.0, epsilon=0.1))
    ]),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
}

resultados_cv = []

# 3. VALIDACIÓN CRUZADA SIN FUGAS (Solo sobre el entrenamiento)
print("Iniciando validación cruzada...")
for nombre, modelo in configuraciones.items():
    # El Pipeline escala y entrena dentro de cada pliegue automáticamente
    scores = cross_validate(modelo, X_train, y_train, cv=validacion,
                            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
                            n_jobs=1)
    
    resultados_cv.append({
        "Modelo": nombre,
        "MAE_medio": -scores["test_neg_mean_absolute_error"].mean(),
        "RMSE_medio": -scores["test_neg_root_mean_squared_error"].mean()
    })

tabla_cv = pd.DataFrame(resultados_cv).sort_values("RMSE_medio")
print("\nComparación en Validación Cruzada:")
print(tabla_cv.to_string(index=False))

# 4. EVALUACIÓN FINAL Y GRÁFICOS
mejor_modelo_nombre = tabla_cv.iloc[0]["Modelo"]
mejor_modelo = configuraciones[mejor_modelo_nombre]

# Ajuste final con todo el entrenamiento
mejor_modelo.fit(X_train, y_train)

# Predicción asegurando el rango 1-5 estrellas exigido por el profesor
predicciones = np.clip(np.round(mejor_modelo.predict(X_test)), 1, 5)

print(f"\nEvaluación Final en Test con {mejor_modelo_nombre}:")
print(f"MAE: {mean_absolute_error(y_test, predicciones):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predicciones)):.3f}")

# Gráfico comparativo de distribuciones (Predicción vs Realidad)
plt.figure(figsize=(10, 5))
plt.hist(y_test, bins=np.arange(1, 7)-0.25, width=0.2, color='#17324D', alpha=0.7, label='Realidad')
plt.hist(predicciones, bins=np.arange(1, 7)+0.05, width=0.2, color='#D6751D', alpha=0.7, label='Predicciones')
plt.title(f"Distribución de Predicciones - {mejor_modelo_nombre}")
plt.xlabel("Estrellas (1-5)")
plt.ylabel("Frecuencia")
plt.legend()
plt.xticks(range(1, 6))
plt.savefig("resultados/distribucion_predicciones.png")
plt.show()