import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Script 2 - Problema 2: Entrenamiento del clasificador de gestos

# Carga y preprocesamiento de datos
# Cargamos los archivos generados con record-dataset.py
DATA_FILE = "rps_dataset.npy"
LABEL_FILE = "rps_labels.npy"
data = np.load(DATA_FILE, allow_pickle=True)
labels = np.load(LABEL_FILE, allow_pickle=True)

# Convertir labels a one-hot encoding (clasificación multiclase)
labels = to_categorical(labels, num_classes=3)
# Convertir data a numpy array (forma: [num_ejemplos, 42])
data = np.array(data)
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# Modelo
# Definición
model = Sequential([
    Dense(64, activation='relu', input_shape=(data.shape[1],)),  
    Dense(64, activation='relu'),                     
    Dense(3, activation='softmax') # capa de salida -> 3 clases
])

# Compilación
optimizer = Adam(learning_rate=0.01)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy', # clasificación multiclase
    metrics=['accuracy']
)

# Entrenamiento
history = model.fit(
    data, 
    labels,
    epochs=50,        
    batch_size=16,    
    validation_split=0.2, # 20% de los datos para validar
    verbose=1
)

# Evaluación
# Gráfica accuracy por épocas
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title("Accuracy por época")
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Gráfica loss por épocas
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title("Loss por época")
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Acuraccy final sobre todo el dataset
y_pred = model.predict(data)
pred_label = np.argmax(y_pred, axis=1)
true_label = np.argmax(labels, axis=1)
accuracy_total = np.sum(pred_label == true_label) / len(true_label)
print(f"\nAccuracy total: {accuracy_total:.2f}")

# Matriz de confusión
class_names = ["piedra", "papel", "tijera"]
num_labels = labels.shape[1]
cm = np.zeros((num_labels, num_labels), dtype=int)
for t, p in zip(true_label, pred_label):
    cm[t, p] += 1
# Gráfico
plt.imshow(cm, cmap='Blues')
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.xticks(range(num_labels), class_names)
plt.ylabel("Real")
plt.yticks(range(num_labels), class_names)
for i in range(num_labels):
    for j in range(num_labels):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.colorbar()
plt.show()

# Precision, recall, f1 score por clase
precision = np.diag(cm) / np.sum(cm, axis=0)
recall = np.diag(cm) / np.sum(cm, axis=1)
f1 = 2 * (precision * recall) / (precision + recall)
for i, name in enumerate(class_names):
    print(f"\nClase {name}:\n* Precision: {precision[i]:.2f}\n* Recall: {recall[i]:.2f}\n* F1 score: {f1[i]:.2f}")

## Guardar modelo entrenado
model.save("rps_model.h5")
print("\nModelo guardado: rps_model.h5")