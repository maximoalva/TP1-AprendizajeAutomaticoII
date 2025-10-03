import os
import numpy as np
import cv2
import mediapipe as mp

# Script 1 - Problema 2: Grabación del dataset de gestos

# Inicializamos MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, # una sola mano
    min_detection_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

# Creamos archivos de salida
DATA_FILE = "rps_dataset.npy"
LABEL_FILE = "rps_labels.npy"
# Si existen datasets previos, los cargamos
if os.path.exists(DATA_FILE) and os.path.exists(LABEL_FILE):
    data = np.load(DATA_FILE).tolist()
    labels = np.load(LABEL_FILE).tolist()
    print(f"\nDataset existente cargado: {len(data)} ejemplos.")
else:
    data, labels = [], []
    print("\nNuevo dataset creado correctamente.")

# Creamos función para extraer landmarks en formato plano
def get_hand_landmarks(results) -> list[float] | None:
    """
    Devuelve los landmarks de la mano en formato plano [x1, y1, x2,  y2, ...].

    Parámetros:
    results: mediapipe.python.solution_base.SolutionOutputs
        Objeto devuelto por hands.process() de MediaPipe.
    Return:
    list[float] | None
        Lista con las coordenadas normalizadas (x, y) de los 21 landmarks
        o None si no se detectó ninguna mano.
    """
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # PODEMOS SACARLO POR HANDS=1
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y]) # solo x e y
        return coords
    return None


# Abrimos la cámara
cap = cv2.VideoCapture(0)
# ret -> booleano; frame -> la imagen capturada en ese momento, array de numpy
while True:
    ret, frame = cap.read() 
    if not ret:
        break
    
    # Convertimos la imagen a RGB para MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Procesamos los resultados
    results = hands.process(img_rgb)
    # Si detecta una mano, dibuja los landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Grabando dataset - Piedra, papel o tijera.", frame)
    
    # Guardar los ejemplos apretando sus respectivas teclas.
    # 'p' -> Piedra -> 0; 'a' -> Papel -> 1; 't' -> Tijeras -> 2; 'q' -> Salir
    key = cv2.waitKey(1) & 0xFF 
    key_map = {ord('p'): 0, ord('a'): 1, ord('t'): 2}
    
    if key in key_map:
        landmarks = get_hand_landmarks(results)
        if landmarks:  # si hay mano detectada
            label = key_map[key]
            data.append(landmarks)
            labels.append(label)
            print(f"¡Ejemplo guardado! Clase: {label}, Total: {len(data)}")
            
            # Duplicado con flip horizontal
            landmarks_flipped = landmarks.copy()
            for i in range(0, len(landmarks_flipped), 2):
                landmarks_flipped[i] = 1.0 - landmarks_flipped[i] # flip x
            data.append(landmarks_flipped)
            labels.append(label)
            print(f"¡Ejemplo invertido guardado! Clase: {label}, Total: {len(data)}")
    # Salir
    elif key == ord('q'):
        break

# Cerramos la cámara
cap.release()
cv2.destroyAllWindows()

# Exportamos dataset en archivos .npy
np.save(DATA_FILE, np.array(data))
np.save(LABEL_FILE, np.array(labels))
print(f"\nDataset guardado: {len(data)} ejemplos en {DATA_FILE} y {LABEL_FILE}")