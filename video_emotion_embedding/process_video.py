import operator
import pickle
from functools import partial
import numpy as np
import cv2
import moviepy.editor as mp
from scipy import ndimage
from scipy.stats import norm

from ahegao_detection_eval_show_video import init_ssd_face_detection, init_ahegao_classifier_atulapra, \
    init_rfcn_face_detection


def annotate_frame(frame, face_detector, emotion_classifier):
    faces, scores = face_detector(frame)
    # draw box around face with maximum area
    font_kwargs = {'fontScale': 1, 'thickness': 1, 'fontFace': cv2.FONT_HERSHEY_DUPLEX}
    for face in faces:
        (x, y, w, h) = face
        offset = 15
        face_img = frame[
                   max(y - offset, 0):min(y + h + offset, frame.shape[0]),
                   max(x - offset, 0): min(x + w + offset, frame.shape[1])].copy()
        # frame = cv2.rectangle(frame, (x - 30, y - 30), (x + w + 30, y + h + 30), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        emotions = emotion_classifier(face_img)

        emotion = max(emotions.items(), key=operator.itemgetter(1))[0]
        prob = emotions[emotion]
        cv2.putText(frame, f'{emotion}: {prob * 100:.0f}%', (x - 10, y - 10), color=(255, 255, 255), **font_kwargs)
        if emotion not in emotions_histogram:
            emotions_histogram[emotion] = 1
        emotions_histogram[emotion] += 1

    if 'Neutral' in emotions_histogram:
        neutral_ratio = emotions_histogram['Neutral'] / sum(emotions_histogram.values())
    else:
        neutral_ratio = 0.
    cv2.putText(frame, f'neutral percentage: {neutral_ratio * 100:.1f}%', (20, 30), color=(255, 55, 55),
                **font_kwargs)

    return frame


def flexing_with_video():
    # uncomment any of the fl* function to use that and comment the rest of them
    video = mp.VideoFileClip('videos/twilight-trailer.mp4')

    # newclip = video.fl_image(lambda x: np.stack((x[:, :, 0], x[:, :, 0], x[:, :, 0]), axis=-1))
    # newclip = video.fl_image(lambda x: np.stack((x[:, :, 0], x[:, :, 1], x[:, :, 0]), axis=-1))
    # newclip = video.fl_image(lambda x: np.stack((x[:, :, 0], np.zeros_like(x[:, :, 0]), np.zeros_like(x[:, :, 0])), axis=-1))
    # newclip = video.fl_image(lambda x: np.stack((x.mean(axis=-1), x[:, :, 1], x[:, :, 2]), axis=-1))
    # newclip = video.fl_image(lambda x: np.stack((x.sum(axis=-1) * 255 // x.sum(axis=-1).max(), x[:, :, 1], x[:, :, 2]), axis=-1))
    # newclip = video.fl_image(lambda x: np.stack((np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1)), x[:, :, 1], x[:, :, 2]), axis=-1))

    def pulse_by_frame(gf, t):
        x = gf(t)
        smoothing = norm.pdf([t % 4, t % 5, t % 6], scale=2) * 5  # multiply by 5 so it scales to 1 at point 0
        x = x * smoothing * 2
        return np.minimum(x, 255)

    # newclip = video.fl_image(lambda x: np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1)))
    newclip = video.fl(pulse_by_frame)
    newclip.write_videofile('videos/twilight-trailer-pulse.mp4')


def main():
    # face_detection_ssd = init_ssd_face_detection()
    face_detection_rfcn = init_rfcn_face_detection()
    emotion_classifier = init_ahegao_classifier_atulapra()
    video = mp.VideoFileClip('videos/twilight-movie-1--lunch-scene.mp4')

    newclip = video.fl_image(
        partial(annotate_frame, face_detector=face_detection_rfcn, emotion_classifier=emotion_classifier),
        apply_to='mask')
    newclip.write_videofile('videos/twilight-movie-1--lunch-scene-rfcn.mp4')
    with open('twilight-movie-1-emotions.rick', 'wb+') as f:
        pickle.dump(emotions_histogram, f)


if __name__ == '__main__':
    emotions_histogram = {}
    main()

# for face detection try https://github.com/the-house-of-black-and-white/hall-of-faces
# or https://cmusatyalab.github.io/openface/models-and-accuracies/
# or https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# for emotion recognition use https://talhassner.github.io/home/publication/2015_ICMI
# or https://github.com/fengju514/Expression-Net
