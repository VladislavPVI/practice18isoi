import cv2
import PIL.Image
import dlib
import numpy as np

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

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):

    # Сраниваем характеристики известеных лиц и нового лица
    # Если значение Евклидова расстояния между дескрипторами меньше 0.6,
    # то считается, что на фотографиях один и тот же человек

    if len(known_face_encodings) == 0:
        face_distance = np.empty((0))
    else:
        face_distance = np.linalg.norm(known_face_encodings - face_encoding_to_check, axis=1)

    return list(face_distance <= tolerance)

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


# Получаем ссылку на вебкамеру #0 (по умолчанию)
video_capture = cv2.VideoCapture(0)

# Получаем характеристики лиц людей, которых необходимо распознавать
putin_image = np.array(PIL.Image.open("putin.jpg"))
putin_face_encoding = face_encodings(putin_image)[0]

yakimov_image = np.array(PIL.Image.open("yakimov.jpg"))
yakimov_face_encoding = face_encodings(yakimov_image)[0]

skatova_image = np.array(PIL.Image.open("skatova.jpg"))
skatova_face_encoding = face_encodings(skatova_image)[0]

my_image = np.array(PIL.Image.open("123.jpg"))
my_face_encoding = face_encodings(my_image)[0]

lesha = np.array(PIL.Image.open("lesha.jpg"))
lesha_face_encoding = face_encodings(lesha)[0]

elena = np.array(PIL.Image.open("elena.jpg"))
elena_face_encoding = face_encodings(elena)[0]

# Создаем массивы с характеристиками и именами тех, кому они принадлежат
known_face_encodings = [
    putin_face_encoding,
    yakimov_face_encoding,
    skatova_face_encoding,
    my_face_encoding,
    lesha_face_encoding,
    elena_face_encoding
]
known_face_names = [
    "Vladimir Putin",
    "Pavel Yakimov",
    "Elena Skatova",
    "Vladislav Pshenin",
    "Alex Raku",
    "Elena Makarova"
]

while True:
    # Извлекаем 1 кадр
    ret, frame = video_capture.read()

    # Ковертируем изображение из BGR (используеся в OpenCV) в RGB
    rgb_frame = frame[:, :, ::-1]

    # Находим все лица на кадре и их характеристики
    face_locationsM = face_locations(rgb_frame)
    face_encodingsM = face_encodings(rgb_frame, face_locationsM)

    # Цикл по всем лицам на кадре
    for (top, right, bottom, left), face_encoding in zip(face_locationsM, face_encodingsM):
        # Сравниваем лицо на изображении и извсетные нам лица
        matches = compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

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