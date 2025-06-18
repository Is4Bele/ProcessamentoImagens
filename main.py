
# ViziGuardian - Detecção de Pessoas Suspeitas com OpenCV + YOLOv8

import cv2
from ultralytics import YOLO
from datetime import datetime
import os

# Inicializar modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")  # Use yolov8n.pt para melhor desempenho em tempo real

# Pasta para salvar imagens suspeitas
if not os.path.exists("detecoes"):
    os.makedirs("detecoes")

# Iniciar webcam
cap = cv2.VideoCapture(0)

# Parâmetros
TEMPO_PARADO = 10  # segundos
registro_pessoas = {}

print("[INFO] Monitoramento iniciado. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.5:  # Pessoa detectada
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Pessoa {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                key = f"{cx}-{cy}"
                now = datetime.now()

                if key not in registro_pessoas:
                    registro_pessoas[key] = now
                elif (now - registro_pessoas[key]).seconds > TEMPO_PARADO:
                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"detecoes/suspeito_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[ALERTA] Pessoa parada detectada e salva em {filename}")
                    del registro_pessoas[key]

    cv2.imshow("ViziGuardian - Monitoramento", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
