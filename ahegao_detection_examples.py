from EmoPy.src.fermodel import FERModel
from pkg_resources import resource_filename
import cv2
import numpy as np


def predict_frame(model: FERModel, image):
    gray_image = image
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, model.target_dimensions, interpolation=cv2.INTER_LINEAR)
    final_image = np.array([np.array([resized_image]).reshape(list(model.target_dimensions) + [model.channels])])
    prediction = model.model.predict(final_image)
    return prediction


# todo: try only face detection and then merge it with emopy
# todo maybe try to compare haarclassifier with some other face detection, and then benchmark models
# todo: maybe try https://github.com/JsFlo/EmotionRecTraining repo
# todo: maybe try https://github.com/gauravtheP/Real-Time-Facial-Expression-Recognition
# todo: prolly clean dataset by detecting and cleaning duels (or not, maybe)
def main():
    target_emotions = ['anger', 'fear', 'surprise', 'calm']
    model = FERModel(target_emotions, verbose=True)

    # prevents opencl usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    cap = cv2.VideoCapture('videos/6614059835535133954.mp4')
    # cap = cv2.VideoCapture('videos/6620049294030277893.mp4')
    # cap = cv2.VideoCapture('videos/6628669426822548741.mp4')
    feelings_faces = [cv2.imread('./emojis/' + emotion + '.png', -1) for index, emotion in enumerate(EMOTIONS)]

    # append the list with the emoji images
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

    while True:
        # Again find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break

        # compute softmax probabilities
        result = predict_frame(model, frame)
        # todo: try and fix prolly
        # if result is not None:
        #     # write the different emotions and have a bar to indicate probabilities for each class
        #     for index, emotion in enumerate(EMOTIONS):
        #         cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        #         cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
        #                       (255, 0, 0), -1)
        #
        #     # find the emotion with maximum probability and display it
        #     maxindex = np.argmax(result[0])
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(frame, EMOTIONS[maxindex], (10, 360), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        #     face_image = feelings_faces[maxindex]
        #
        #     for c in range(0, 3):
        #         # The shape of face_image is (x,y,4). The fourth channel is 0 or 1. In most cases it is 0, so, we assign the roi to the emoji.
        #         # You could also do: frame[200:320,10:130,c] = frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
        #         frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130,
        #                                                                                           c] * (
        #                                                 1.0 - face_image[:, :, 3] / 255.0)
        #
        # if len(faces) > 0:
        #     # draw box around face with maximum area
        #     max_area_face = faces[0]
        #     for face in faces:
        #         if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
        #             max_area_face = face
        #     face = max_area_face
        #     (x, y, w, h) = max_area_face
        #     frame = cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

        cv2.imshow('Video', cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
