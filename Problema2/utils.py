# Utils - Problema 2: Funciones útiles para todos los scripts

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