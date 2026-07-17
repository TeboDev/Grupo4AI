# -*- coding: utf-8 -*-
"""
@author: IVAN
"""

# Naive Bayes for Diabetes (Hypoglycemia prediction)

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

print("--- 1. CARGA DE DATOS PREPROCESADOS ---")
# Ruta robusta: busca el CSV en varias ubicaciones posibles
directorio_actual = os.path.dirname(os.path.abspath(__file__))

rutas_posibles = [
    os.path.join(directorio_actual, '..', '1. Preprocesamiento', 'diabetes_preprocessed.csv'),
    os.path.join(directorio_actual, '..', 'diabetes_preprocessed.csv'),
    'diabetes_preprocessed.csv',
]

df_ml = None
for ruta in rutas_posibles:
    if os.path.exists(ruta):
        df_ml = pd.read_csv(ruta)
        print(f"Datos cargados exitosamente de: {ruta}")
        break

if df_ml is None:
    print("Error: No se encontró 'diabetes_preprocessed.csv'. Ejecuta 'preprocess_data.py' primero.")
    exit()

# Separamos las características (X) y el objetivo (y)
# Objetivo: predecir si hubo hipoglucemia (1) o no (0)
X = df_ml.drop(columns=['Hubo_Hipoglucemia']).values
y = df_ml['Hubo_Hipoglucemia'].values
feature_names = df_ml.drop(columns=['Hubo_Hipoglucemia']).columns.tolist()

print(f"Dimensiones de X (Características): {X.shape}")
print(f"Dimensiones de y (Objetivo): {y.shape}\n")

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normaliza los datos para que todas las características tengan una escala similar
# (el dataset ya fue preprocesado con MinMaxScaler, pero re-escalamos por consistencia)
scaler = MinMaxScaler(feature_range=(0, 1))  # [0, 1]
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crea el modelo Naive Bayes (Gaussiano)
# GaussianNB asume que los datos continuos siguen una distribución aproximadamente normal.
model = GaussianNB()

# Variables categóricas discretas usarían CategoricalNB (no normalizado):
# from sklearn.naive_bayes import CategoricalNB
# model = CategoricalNB()

model.fit(X_train, y_train)

# Realiza predicciones usando el conjunto de prueba
y_pred = model.predict(X_test)

# Muestra el informe de evaluación del modelo entrenado
print("--- 5. EVALUACIÓN DEL MODELO ---")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusión:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: \n", cm)

# Gráfica de la matriz de confusión
plt.figure(figsize=(8, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Hipoglucemia', 'Hipoglucemia'],
            yticklabels=['No Hipoglucemia', 'Hipoglucemia'])
plt.title('Matriz de Confusión - Naive Bayes (Diabetes)')
plt.xlabel('Predicción', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.tight_layout()
plt.show()

# Exactitud:
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("accuracy: ", acc)

# Sensibilidad:
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred, zero_division=0)
print("recall: ", recall)

# Precisión:
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred, zero_division=0)
print("precision: ", precision)

# Especificidad
# 'specificity' is just a special case of 'recall'.
# specificity is the recall of the negative class
specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
print("specificity: ", specificity)

# Puntuación F1:
from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, zero_division=0)
print("f1 score: ", f1)

# Área bajo la curva:
from sklearn.metrics import roc_auc_score
try:
    auc = roc_auc_score(y_test, y_pred)
    print("auc: ", auc)
except ValueError:
    print("AUC: No se puede calcular (solo una clase en y_test)")
    auc = None

# Curva ROC
from sklearn.metrics import roc_curve
if auc is not None:
    plt.figure()
    lw = 2
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
             label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Naive Bayes (Diabetes)')
    plt.legend(loc="lower right")
    plt.show()

# R Score (R^2 coefficient of determination)
from sklearn.metrics import r2_score
R = r2_score(y_test, y_pred)
print("R2: ", R)

# --- GRÁFICO ADICIONAL: Importancia de características (varianza de clases) ---
# GaussianNB estima la varianza por clase; podemos visualizar la diferencia de medias
import numpy as np

means_class0 = model.theta_[0]   # medias para clase 0 (sin hipoglucemia)
means_class1 = model.theta_[1]   # medias para clase 1 (con hipoglucemia)
diff_means = means_class1 - means_class0

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['tomato' if d > 0 else 'steelblue' for d in diff_means]
ax.barh(feature_names, diff_means, color=colors)
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Diferencia de Medias por Clase - Naive Bayes (Diabetes)\n'
             '(Positivo → mayor en Hipoglucemia; Negativo → mayor en No Hipoglucemia)')
ax.set_xlabel('Diferencia de Medias (Clase 1 − Clase 0)')
plt.tight_layout()
plt.show()

# Guardar el modelo a un archivo
import joblib
ruta_modelo = os.path.join(directorio_actual, 'NB_model_diabetes.pkl')
joblib.dump(model, ruta_modelo)
print(f"\nModelo guardado exitosamente como '{ruta_modelo}'")

# Cargar el modelo desde el archivo directamente sin reentrenar
loaded_model = joblib.load(ruta_modelo)
# Hacer predicciones con el modelo cargado
y_pred = loaded_model.predict(X_test)
