# Taller de Machine Learning - Semana 1: Regresión Lineal y Clasificación de Churn

Este repositorio contiene los informes, notebooks y plantillas de código para el taller de la **Semana 1** del módulo de Machine Learning (Maestría en Ciencia de Datos). El objetivo de este taller es dominar los conceptos de **Regresión Lineal** (regularización L1/L2 y aprendizaje online por mini-lotes) y la **Clasificación de Churn** en telecomunicaciones (manejo de datos desbalanceados con SMOTE, Pesos de Clase y PCA).

---

## 📂 Archivos del Taller e Informes Finales

| Entregable / Notebook | Descripción | Archivos del Repositorio |
| :--- | :--- | :--- |
| **Informe de Taller 01** | Reporte detallado de Regresión Lineal con Optuna (Nancy Altamirano, Kevin Viteri) | [Informe_Taller01_Regresion_Lineal.pdf](Informe_Taller01_Regresion_Lineal.pdf) |
| **Informe de Taller 02** | Reporte detallado de Clasificación de Churn con SMOTE, Pesos y PCA | [Informe_Taller02_Clasificacion_Churn.pdf](Informe_Taller02_Clasificacion_Churn.pdf) |
| **Notebook Churn** | Código ejecutado con análisis y visualización de predicción de Churn | [Notebook Churn (ipynb)](Planteamiento_Clasificación_Churn_Telecomunicaciones_Logistic_Regression.ipynb) <br> [Notebook Churn (PDF)](Planteamiento_Clasificación_Churn_Telecomunicaciones_Logistic_Regression.pdf) |

---

## 🚚 Entregas Realizadas

1. **Archivos PDF de los Informes** con sustento teórico, metodológico y evidencia gráfica.
2. **Notebooks Completamente Ejecutados** con todos sus gráficos de salida y análisis integrados.

---

## 1. Guía de Trabajo: Regresión Lineal (`linear_regression.ipynb`)

El laboratorio se dividió en dos secciones principales: **Pipelines y Regularización** (Ridge/Lasso) sobre el dataset de vivienda de California, y **Aprendizaje Online (Online Learning)** con descenso de gradiente estocástico (`SGDRegressor`) sobre el *Million Song Dataset* (MSD).

### 🛠️ Ejercicios Resueltos:

#### Parte 1: Pipelines y Regularización
- **[x] Ejercicio 1.1 — Efecto del Grado Polinomial (Básico)**
  - Comparar los grados polinomiales de entrada `[1, 2, 3]`.
  - Construir un Pipeline para cada grado, realizar `GridSearchCV` / `Optuna` y registrar el mejor RMSE de validación cruzada y el mejor $\alpha$.
  - **Pregunta teórica:** ¿Un grado polinomial más alto siempre mejora el rendimiento en prueba? Explicado en el informe.
- **[x] Ejercicio 1.2 — Ridge vs. Lasso**
  - Comparar el RMSE en el conjunto de prueba obtenido por Ridge y Lasso.
  - Contar y comparar cuántos coeficientes del polinomio de grado 2 se reducen exactamente a cero en cada modelo (Lasso zerificó 49/164 coeficientes).

#### Parte 2: Aprendizaje Incremental (Online Learning) con `SGDRegressor`
- **[x] Ejercicio 2.1 — Implementar el Bucle de Entrenamiento por Mini-Lotes (Básico)**
  - Registrar el RMSE de cada chunk procesado usando `partial_fit` e incrementalmente el escalador.
- **[x] Ejercicio 2.2 — Graficar la Curva de Convergencia (Básico)**
  - Graficar el RMSE vs. el índice del chunk.
- **[x] Ejercicio 2.3 — Entrenamiento Multi-Época (Intermedio)**
  - Entrenar `SGDRegressor` durante 10 épocas mezclando los datos y graficar los RMSEs de entrenamiento y test.
- **[x] Ejercicio 2.4 — Exploración de la Tasa de Aprendizaje (Clave)**
  - Comparar tasas `eta0` en `[0.0001, 0.001, 0.01, 0.1, 1.0]`. Identificar tasas divergentes ($\eta_0 \ge 0.1$).
- **[x] Ejercicio 2.5 — Programación de Tasas de Aprendizaje (Schedules - Avanzado)**
  - Evaluar esquemas: `constant`, `optimal` e `invscaling`.

---

## 2. Guía de Trabajo: Clasificador Churn (`Planteamiento_Clasificación_Churn_Telecomunicaciones_Logistic_Regression.ipynb`)

El objetivo fue construir y evaluar un modelo de Regresión Logística optimizado para predecir la pérdida de clientes (`Churn`), contrarrestando el fuerte desbalance de clases para maximizar el **Recall**.

### 🛠️ Actividades Resueltas:

- **[x] Actividad 1: Optimización de Regularización y Evaluación de Métrica**
  - Búsqueda en rejilla de $\lambda$ óptimo ($\lambda = 10000$ / C = 0.0001) con CV=5 sobre el conjunto submuestreado.
  - Generación de la curva de complejidad (Recall vs. Lambda).
  - **Pregunta:** Justificación del Recall como métrica comercial crítica para reducir Falsos Negativos.
- **[x] Actividad 2: Tratamiento del Desbalance mediante SMOTE y Pesos de Clase**
  - Implementación de un `ImbPipeline` aplicando sobremuestreo sintético (SMOTE) y penalización de pesos de clase inversamente proporcionales en la Regresión Logística.
  - Comparación de matrices de confusión frente al submuestreo de la Actividad 1.
- **[x] Actividad 3: Integración de SMOTE, Pesos de Clase y PCA**
  - Pipeline completo: `PolynomialFeatures` -> `StandardScaler` -> `PCA` (95% de varianza explicada, 114 componentes) -> `SMOTE` -> `LogisticRegression` con pesos de clase.
  - Obtención de la sensibilidad más alta del estudio (**Recall en prueba = 91.44%**).
- **[x] Conclusiones**
  - Análisis del compromiso (trade-off) entre exactitud y recall y justificación de recomendaciones prácticas de negocio según el presupuesto comercial de la empresa.
