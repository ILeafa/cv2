import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Загрузка предобученной модели YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Загрузка изображения
image_path = "photo.jpg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Подготовка изображения для сети
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Инициализация списков для хранения данных обнаружения
class_ids = []
confidences = []
boxes = []

# Обработка каждого обнаружения
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if class_id == 0 and confidence > 0.5:  # 0 - ID для 'person' в COCO dataset
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Применение Non-MaxSuppression для уменьшения перекрывающихся прямоугольников
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Отрисовка прямоугольников и подсчет людей
people_count = len(indexes)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Отображение количества людей на изображении
cv2.putText(image, f"People count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Определение групп людей
centers = [(int((box[0] + box[0] + box[2]) / 2), int((box[1] + box[1] + box[3]) / 2)) for box in boxes if box in [boxes[i] for i in indexes]]
if len(centers) > 1:
    dist_matrix = cdist(centers, centers, 'euclidean')
    groups = []
    for i, center in enumerate(centers):
        group = [j for j in range(len(centers)) if dist_matrix[i, j] < 100]
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

# Сохранение изображения с результатами
output_path = "output_image.png"
cv2.imwrite(output_path, image)