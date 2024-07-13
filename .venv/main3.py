import cv2
from ultralytics import YOLO

# Загрузка моделей YOLOv8
model_people = YOLO('yolov8n.pt')  # модель для обнаружения людей
model_helmets = YOLO('yolov8.pt')  # модель для обнаружения касок

# Загрузка изображения
image_path = 'image.jpg'
image = cv2.imread(image_path)

# Выполнение предсказаний
results_people = model_people(image)
results_helmets = model_helmets(image)

# Списки для хранения координат людей и касок
people_boxes = []
helmet_boxes = []

# Счетчики
people_with_helmets = 0
people_without_helmets = 0

# Обработка результатов для людей
for r in results_people:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 0:  # 0 - ID для 'person' в COCO dataset
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            people_boxes.append((x1, y1, x2, y2))

# Обработка результатов для касок
for r in results_helmets:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Уменьшение размера рамки каски на 20%
        width = x2 - x1
        height = y2 - y1
        reduction_factor = 0.2
        x1 += int(width * reduction_factor / 2)
        y1 += int(height * reduction_factor / 2)
        x2 -= int(width * reduction_factor / 2)
        y2 -= int(height * reduction_factor / 2)
        helmet_boxes.append((x1, y1, x2, y2))

# Обработка результатов для людей
for r in results_people:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 0:  # 0 - ID для 'person' в COCO dataset
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Рисование прямоугольника вокруг человека
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Обработка результатов для касок
for r in results_helmets:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Рисование прямоугольника вокруг каски
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Проверка, есть ли каска внутри рамки человека
for (px1, py1, px2, py2) in people_boxes:
    person_has_helmet = False
    for (hx1, hy1, hx2, hy2) in helmet_boxes:
        if hx1 >= px1 and hy1 >= py1 and hx2 <= px2 and hy2 <= py2:
            person_has_helmet = True
            break

    if person_has_helmet:
        people_with_helmets += 1
        # Рисование прямоугольника вокруг человека в каске
        # cv2.rectangle(image, (px1, py1), (px2, py2), (0, 255, 0), 2)
    else:
        people_without_helmets += 1
        # Рисование прямоугольника вокруг человека без каски
        # cv2.rectangle(image, (px1, py1), (px2, py2), (0, 0, 255), 2)

# Добавление текста с количеством людей в касках и без касок

text_with_helmets = f"People with helmets: {people_with_helmets}"
text_without_helmets = f"People without helmets: {people_without_helmets}"
cv2.putText(image, text_with_helmets, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(image, text_without_helmets, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Сохранение результата
cv2.imwrite('output_image3.png', image)