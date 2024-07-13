import cv2
from ultralytics import YOLO

# Загрузка моделей YOLOv8 для обнаружения людей
model_people = YOLO('yolov8n.pt')

# Загрузка изображения
image_path = 'image.jpg'
image = cv2.imread(image_path)

# Выполнение предсказаний
results_people = model_people(image)

# Счетчики
people_count = 0

# Обработка результатов для людей
for r in results_people:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 0:  # 0 - ID для 'person' в COCO dataset
            people_count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Рисование прямоугольника вокруг человека
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Добавление текста с количеством людей
text_people = f"Number of people: {people_count}"
cv2.putText(image, text_people, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Сохранение результата
cv2.imwrite('output_image.png', image)