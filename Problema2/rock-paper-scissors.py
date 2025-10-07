import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import get_hand_landmarks

# Script 3 - Problema 2: Prueba del sistema completo

# Cargar modelo entrenado
model = load_model("rps_model.h5")

# Inicializamos MediaPipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, # una sola mano
    min_detection_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils

# Especificamos los nombres de las clases para etiquetar en tiempo real
class_names = ["piedra", "papel", "tijera"]

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
    
        landmarks = get_hand_landmarks(results)
        if landmarks: # si hay mano detectada
            pred = model.predict(np.array([landmarks])) # Realiza la predicción con los landmarks actuales
            pred_i = np.argmax(pred) # Índice del gesto que obtuvo mayor probabilidad
            pred_class = class_names[pred_i] # Nombre de la clase obtenida
            prob = np.max(pred) # Probabilidad de la clase obtenida
            # Mostrar en tiempo real los resultados
            cv2.putText(
                frame,
                f"{pred_class} ({prob*100:.1f}%)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imshow("Detección de gesto - Piedra, papel o tijera.", frame)
    
    # Salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cerramos la cámara
cap.release()
cv2.destroyAllWindows()