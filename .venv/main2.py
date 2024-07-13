import cv2
from ultralytics import YOLO
from scipy.spatial.distance import cdist

# Загрузка моделей YOLOv8 для обнаружения людей
model_people = YOLO('yolov8n.pt')

# Загрузка изображения
image_path = 'image.jpg'
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Выполнение предсказаний
results_people = model_people(image)

# Списки для хранения данных обнаружения
boxes = []

# Счетчики
people_count = 0

# Обработка результатов для людей
for r in results_people:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # 0 - ID для 'person' в COCO dataset
            people_count += 1
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            boxes.append([x1, y1, x2 - x1, y2 - y1])

            # Рисование прямоугольника вокруг человека
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Определение центров обнаруженных людей
centers = [(int((box[0] + box[0] + box[2]) / 2), int((box[1] + box[1] + box[3]) / 2)) for box in boxes]

# Определение групп людей
if len(centers) > 1:
    dist_matrix = cdist(centers, centers, 'euclidean')
    groups = []
    for i, center in enumerate(centers):
        group = [j for j in range(len(centers)) if dist_matrix[i, j] < 50]
        if len(group) > 1:
            groups.append(group)

    # Отрисовка прямоугольников вокруг групп
    for group in groups:
        group_boxes = [boxes[i] for i in group]
        x_min = min([box[0] for box in group_boxes])
        y_min = min([box[1] for box in group_boxes])
        x_max = max([box[0] + box[2] for box in group_boxes])
        y_max = max([box[1] + box[3] for box in group_boxes])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, f"Group: {len(group)}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Добавление текста с количеством людей
text_people = f"Number of people: {people_count}"
cv2.putText(image, text_people, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Сохранение результата
cv2.imwrite('output_image2.png', image)