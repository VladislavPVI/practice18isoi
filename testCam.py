import re
import cv2
import PIL.Image
import dlib
import numpy as np
import os
import pickle
from sklearn import neighbors

face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
pose_predictor_5_point = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

def _rect_to_css(rect):

    # Преобразование dlib-объекта 'rect' к виду (top, right, bottom, left)
      return rect.top(), rect.right(), rect.bottom(), rect.left()

def _trim_css_to_bounds(css, image_shape):

    # Проверяем, находится ли наш квадрат в пределах изображаения
       return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def face_locations(img):

    # Возвращет позицию квадрата, внутри которого находятся лица
    return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img)]

def _css_to_rect(css):

    # Обратная функции_rect_to_css
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def _raw_face_locations(img):

    # Возвращет массив объектов 'rect' с координатами лиц
    # Второй аргумент метода показывает, во сколько раз необходимо увеличить изображение
    # Увеличение числа позволяет найти маленькие лица на изображении

    return face_detector(img, 1)

def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def face_encodings(face_image, known_face_locations=None, num_jitters=1):

    # Возвращет 128 характеристик для каждого лица
    # num_jitters - показывает, сколько раз нужно считать характеристики лица
    # Чем число больше, тем точнее, но медленнее

    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model="small")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def train(train_dir, model_save_path=None, n_neighbors=None):

    # массив характеристик
    X = []
    # массив имен
    y = []

    # Цикл по всем папкам тренировочного сета
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Цикл по всем изображениям папки
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = np.array(PIL.Image.open(img_path))
            face_bounding_boxes = face_locations(image)

            if len(face_bounding_boxes) != 1:
                # Если на изображении нет лиц (или их слишком много), пропускаем данное ихображение
                print("Изображение {} не подходит для тренинга: {}".format(img_path, "лиц нет" if len(face_bounding_boxes) < 1 else "найдено больше, чем одно лицо"))
            else:
                # Добавляем харк-ки найденного лица
                X.append(face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Определяем сколько соседей будем использовать для KNN-классификатора
    if n_neighbors is None:
        n_neighbors = int(round(np.sqrt(len(X))))
        print("Число соседей выбрано автоматически:", n_neighbors)

    # Создаем и треним классификатор
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)

    # Сохраняем классификатор
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

if __name__ == "__main__":

    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # Получаем ссылку на вебкамеру #0 (по умолчанию)
    video_capture = cv2.VideoCapture(0)


    knn_clf = None
    model_path="trained_knn_model.clf"
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    process_this_frame = True
    names = []
    face_locationsM = []

    while True:
        # Извлекаем 1 кадр
        ret, frame = video_capture.read()

        # Уменьшаем кадр в 4 раза для ускорения
        little_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Ковертируем изображение из BGR (используеся в OpenCV) в RGB
        little_rgb_frame = little_frame[:, :, ::-1]

        if process_this_frame:

            # Находим все лица на кадре и их характеристики
            face_locationsM = face_locations(little_rgb_frame)
            if face_locationsM:
                face_encodingsM = face_encodings(little_rgb_frame, face_locationsM)

                closest_distances = knn_clf.kneighbors(face_encodingsM, n_neighbors=1)
                names = []
                faces = knn_clf.predict(face_encodingsM)
                for i in range(len(face_locationsM)):
                    name = "Unknown"
                    if closest_distances[0][i][0] <= 0.6:
                        name = faces[i]
                    names.append(name)

        process_this_frame = not process_this_frame
        # Цикл по всем лицам на кадре
        for (top, right, bottom, left), name in zip(face_locationsM, names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Рисуем квадарат вокург лица
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Подписываем найденное лицо
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        # Нажми 'q' для выхода!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Закрываем потоки вебкамеры
    video_capture.release()
    cv2.destroyAllWindows()