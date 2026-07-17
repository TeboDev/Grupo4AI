# -*- coding: utf-8 -*-
"""
@author: IVAN
"""

# RNA for Diabetes (Hypoglycemia prediction)

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("--- 1. CARGA DE DATOS PREPROCESADOS ---")
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
X = df_ml.drop(columns=['Hubo_Hipoglucemia']).values
y = df_ml['Hubo_Hipoglucemia'].values

# Divide el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Normaliza los datos
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n--- 4. ENTRENAMIENTO DEL MODELO RNA ---")
# Crea y entrena el modelo RNA (MLP)
model = Sequential()
# Diabetes tiene 5 variables predictoras, ajustamos input_dim=5
model.add(Dense(10, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) # salida binaria
model.summary()

opt = Adam(learning_rate=1e-2)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Configurar early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# Realiza predicciones usando el conjunto de prueba
y_pred = model.predict(X_test)

# Convierte las salidas en etiquetas binarias (0 o 1)
y_pred = (y_pred > 0.5)

print("\n--- 5. EVALUACIÓN DEL MODELO ---")
# Muestra el informe de evaluación del modelo entrenado
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusión:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix: \n", cm)
# gráfica cm
plt.figure(figsize=(8,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.title('Matriz de Confusión - RNA (Diabetes)')
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
    print("AUC: No se puede calcular")
    auc = None

# Curva ROC
from sklearn.metrics import roc_curve
if auc is not None:
    plt.figure()
    lw = 2
    plt.plot(roc_curve(y_test, y_pred)[0], roc_curve(y_test, y_pred)[1], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' %auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - RNA (Diabetes)')
    plt.legend(loc="lower right")
    plt.show()

# R Score
from sklearn.metrics import r2_score
R = r2_score(y_test, y_pred)
print("R2: ", R)

# Curvas de aprendizaje
plt.title('Loss / binary_crossentropy - RNA (Diabetes)')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy - RNA (Diabetes)')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

# Guardar el modelo
model.save(os.path.join(directorio_actual, 'RNA_model_diabetes.h5'))
print("Modelo guardado exitosamente.")
