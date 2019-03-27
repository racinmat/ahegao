import progressbar
from EmoPy.src.fermodel import FERModel
from docutils.nodes import thead
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import VideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from pkg_resources import resource_filename
import cv2
import os.path as osp
import numpy as np
import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from time import time
from hof.face_detectors import RfcnResnet101FaceDetector, SSDMobileNetV1FaceDetector, FasterRCNNFaceDetector, \
    YOLOv2FaceDetector
from moviepy.video.io.bindings import mplfig_to_npimage
from Emotion_detection.Keras.kerasmodel import build_model
from model import EMR
import matplotlib.animation as manimation

VIDEOS_DIR = 'videos'


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def np_append(arr_1, arr_2):
    if len(arr_1.shape) == 1:
        arr_1 = arr_1[np.newaxis, :]
    arr_1 = np.append(arr_1, arr_2, axis=None if arr_1.shape[1] == 0 else 0)
    if len(arr_1.shape) == 1:
        arr_1 = arr_1[np.newaxis, :]
    return arr_1


def visualize_and_eval(video_name, face_detector, ahegao_classifier=None, output_file=None):
    enable_ahegao_classification = ahegao_classifier is not None

    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(osp.join(VIDEOS_DIR, video_name))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_faces_probs = 3
    if output_file is None:
        plt.ion()
    fig = plt.figure(figsize=(15, 8))
    ax0 = plt.subplot2grid((2, 2), (0, 1))  # number of faces detected
    ax1 = plt.subplot2grid((2, 2), (1, 1))  # showing emotions distribution or faces probs
    ax2 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)  # showing image
    axarr = [ax0, ax1, ax2]
    plt.tight_layout()
    axarr[0].set_title('num faces detected')
    face_line, = axarr[0].plot([], [], 'r-')
    face_probs_lines = []
    if enable_ahegao_classification:
        axarr[1].stackplot([], [])
    else:
        for i in range(max_faces_probs):
            face_probs_line, = axarr[1].plot([], [], 'r-')
            face_probs_lines.append(face_probs_line)
        axarr[1].set_ylim(-0.05, 1.05)
        axarr[1].yaxis.grid(True)
    im = axarr[2].imshow(np.zeros((height, width)))
    axarr[2].grid(False)
    axarr[2].axis('off')

    i = 0
    face_data_x = []
    face_data_y = []
    emotion_data_x = []
    emotion_data_y = np.empty(0)
    face_probs_x = []
    face_probs_y = np.empty((0, 3))
    j = 0

    if output_file is None:
        def update_face_probs(face_probs_x, face_probs_y):
            for k, face_probs_line in enumerate(face_probs_lines):
                face_probs_line.set_xdata(face_probs_x)
                face_probs_line.set_ydata(face_probs_y[:, k])

        def update_face_line(face_data_x, face_data_y):
            face_line.set_xdata(face_data_x)
            face_line.set_ydata(face_data_y)
            # update x and ylim to show all points:
            axarr[0].set_xlim(min(face_data_x) - 0.5, max(face_data_x) + 0.5)
            axarr[0].set_ylim(min(face_data_y) - 0.5, max(face_data_y) + 0.5)

        should_stop = False
        while not should_stop:
            should_stop, emotion_data_x, emotion_data_y, face_probs_x, face_probs_y, i, j = process_frame(
                ahegao_classifier, axarr, cap, emotion_data_x, emotion_data_y, enable_ahegao_classification,
                face_data_x, face_data_y, face_detector, face_probs_x, face_probs_y, i, im, j, max_faces_probs,
                update_face_probs, update_face_line)
            plt.draw()
            plt.pause(0.0001)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        widgets = [progressbar.Percentage(), ' ', progressbar.Counter(), ' ', progressbar.Bar(), ' ',
                   progressbar.FileTransferSpeed()]
        pbar = progressbar.ProgressBar(widgets=widgets, max_value=frame_count).start()

        def update_face_probs(face_probs_x, face_probs_y):
            axarr[1].clear()
            for k, face_probs_line in enumerate(face_probs_lines):
                axarr[1].plot(face_probs_x, face_probs_y[:, k], 'r-')
            axarr[1].set_ylim(-0.05, 1.05)
            axarr[1].yaxis.grid(True)

        def update_face_line(face_data_x, face_data_y):
            axarr[0].clear()
            axarr[0].set_title('num faces detected')
            axarr[0].plot(face_data_x, face_data_y, 'r-')
            axarr[0].set_xlim(min(face_data_x) - 0.5, max(face_data_x) + 0.5)
            axarr[0].set_ylim(min(face_data_y) - 0.5, max(face_data_y) + 0.5)

        # while not should_stop:
        def make_frame(t):
            nonlocal emotion_data_x, emotion_data_y, face_probs_x, face_probs_y, i, j
            pbar.update(i)
            should_stop, emotion_data_x, emotion_data_y, face_probs_x, face_probs_y, i, j = process_frame(
                ahegao_classifier, axarr, cap, emotion_data_x, emotion_data_y, enable_ahegao_classification,
                face_data_x, face_data_y, face_detector, face_probs_x, face_probs_y, i, im, j, max_faces_probs,
                update_face_probs, update_face_line)
            return mplfig_to_npimage(fig)
        pbar.finish()
        orig_audio = AudioFileClip(osp.join(VIDEOS_DIR, video_name))
        animation = VideoClip(make_frame, duration=duration)
        animation.set_audio(orig_audio)
        animation.write_videofile(output_file, fps=fps)

    cap.release()
    cv2.destroyAllWindows()


def process_frame(ahegao_classifier, axarr, cap, emotion_data_x, emotion_data_y, enable_ahegao_classification,
                  face_data_x, face_data_y, face_detector, face_probs_x, face_probs_y, i,
                  im, j, max_faces_probs, update_face_probs, update_face_line):
    ret, frame = cap.read()
    if not ret:
        return True, emotion_data_x, emotion_data_y, face_probs_x, face_probs_y, i, j

    faces, scores = face_detector(frame)
    # draw box around face with maximum area
    for face in faces:
        (x, y, w, h) = face
        offset = 15
        face_img = frame[
                   max(y - offset, 0):min(y + h + offset, frame.shape[0]),
                   max(x - offset, 0): min(x + w + offset, frame.shape[1])].copy()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if not enable_ahegao_classification:
            face_probs_x.append(i)
            scores_arr = np.zeros((1, max_faces_probs))
            scores_arr[:, :len(scores[:max_faces_probs])] = scores[:max_faces_probs]
            face_probs_y = np_append(face_probs_y, scores_arr)
            update_face_probs(face_probs_x, face_probs_y)

            axarr[1].set_xlim(min(face_probs_x) - 0.5, max(face_probs_x) + 0.5)
            continue

        emotions = ahegao_classifier(face_img)

        for k, (emotion, prob) in enumerate(emotions.items()):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f'{emotion}: {prob * 100:.2f}%', (10, 35 + k * 30), font, fontScale=1,
                        color=(255, 255, 255), thickness=2)

        emotion_data_x.append(j)
        emotion_values = np.array(list(emotions.values()))[np.newaxis, :]
        emotion_data_y = np_append(emotion_data_y, emotion_values)

        axarr[1].clear()
        axarr[1].set_title('emotions detected')
        axarr[1].stackplot(emotion_data_x, emotion_data_y.T, labels=list(emotions.keys()))
        # axarr[1].set_ylim(0, 1)
        axarr[1].legend(loc='upper left')

        j += 1
    face_data_x.append(i)
    face_data_y.append(len(faces))
    update_face_line(face_data_x, face_data_y)
    im.set_data(cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB))
    i += 1
    return False, emotion_data_x, emotion_data_y, face_probs_x, face_probs_y, i, j


def init_rfcn_face_detection():
    min_confidence = 0.92  # found empirically by visualizing probabilities
    iou_threshold = 0.2
    rfcn_face_detector = RfcnResnet101FaceDetector(min_confidence=min_confidence)
    graph = rfcn_face_detector.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    # added non max suppression
    with graph.as_default():
        # Each box represents a part of the image where a particular object was detected.
        boxes = graph.get_tensor_by_name('detection_boxes:0')[0]
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = graph.get_tensor_by_name('detection_scores:0')[0]
        classes = graph.get_tensor_by_name('detection_classes:0')[0]
        num_detections = tf.to_int32(graph.get_tensor_by_name('num_detections:0')[0])

        selected_indices = tf.image.non_max_suppression(boxes, scores, num_detections, iou_threshold, min_confidence)
        classes_selected = tf.gather(classes, selected_indices)
        scores_selected = tf.gather(scores, selected_indices)
        boxes_selected = tf.gather(boxes, selected_indices)

    def face_detection_rfcn(frame):
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        boxes, scores, classes = rfcn_face_detector.sess.run([boxes_selected, scores_selected, classes_selected],
                                                             feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        boxes = boxes[scores >= rfcn_face_detector.min_confidence, :]
        scores = scores[scores >= rfcn_face_detector.min_confidence]
        # ymin, xmin, ymax, xmax = box
        faces = np.zeros_like(boxes)
        faces[:, 1] = boxes[:, 0] * frame.shape[0]
        faces[:, 0] = boxes[:, 1] * frame.shape[1]
        faces[:, 3] = (boxes[:, 2] - boxes[:, 0]) * frame.shape[0]
        faces[:, 2] = (boxes[:, 3] - boxes[:, 1]) * frame.shape[1]
        faces = faces.astype(np.int32)

        scores = scores[faces[:, 2] * faces[:, 3] > 1000]
        faces = faces[faces[:, 2] * faces[:, 3] > 1000, :]  # exclude too small faces (1000px and less)
        return faces, scores

    return face_detection_rfcn


def init_ssd_face_detection():
    path_to_ckpt = 'tensorflow_face_detection/model/frozen_inference_graph_face.pb'
    path_to_ckpt = osp.join(osp.dirname(osp.abspath(__file__)), path_to_ckpt)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=detection_graph, config=config)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        def face_detection_ssd(frame):
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            boxes, scores = sess.run([boxes_tensor, scores_tensor], feed_dict={image_tensor: image_np_expanded})
            # to format returned by opencv
            threshold = 0.5
            boxes = boxes[scores > threshold, :]
            scores = scores[scores > threshold]
            # ymin, xmin, ymax, xmax = box
            faces = np.zeros_like(boxes)
            faces[:, 1] = boxes[:, 0] * frame.shape[0]
            faces[:, 0] = boxes[:, 1] * frame.shape[1]
            faces[:, 3] = (boxes[:, 2] - boxes[:, 0]) * frame.shape[0]
            faces[:, 2] = (boxes[:, 3] - boxes[:, 1]) * frame.shape[1]
            faces = faces.astype(np.int32)
            return faces, scores

        return face_detection_ssd


def init_haar_cascade_detector():
    facecasc = cv2.CascadeClassifier('Emotion/models/haarcascade_frontalface_default.xml')

    def face_detection_haar_cascade(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        return faces, np.empty_like(faces) if len(faces) == 0 else np.ones(faces.shape[0])

    return face_detection_haar_cascade


def init_ahegao_classifier_emopy():
    target_emotions = ['anger', 'fear', 'surprise', 'calm']
    model = FERModel(target_emotions, verbose=True)

    def ahegao_classification_emopy(frame):
        gray_image = frame
        if len(frame.shape) > 2:
            gray_image = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, model.target_dimensions, interpolation=cv2.INTER_CUBIC)
        final_image = np.expand_dims(np.expand_dims(resized_image, -1), 0)
        prediction = model.model.predict(final_image)[0]
        # normalized_prediction = prediction / sum(prediction)
        normalized_prediction = softmax(prediction)
        return {emotion: normalized_prediction[i] for i, emotion in enumerate(model.emotion_map.keys())}

    return ahegao_classification_emopy


# from repo https://github.com/atulapra/Emotion-detection.git
def init_ahegao_classifier_atulapra():
    network = build_model()
    path_to_h5 = 'Emotion_detection/Keras/model.h5'
    path_to_h5 = osp.join(osp.dirname(osp.abspath(__file__)), path_to_h5)
    network.load_weights(path_to_h5)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    def ahegao_classification_atulapra(frame):
        gray_image = frame
        if len(frame.shape) > 2:
            gray_image = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (48, 48), interpolation=cv2.INTER_CUBIC)
        resized_image = np.expand_dims(np.expand_dims(resized_image, -1), 0)
        resized_image = resized_image / 255.
        prediction = network.predict(resized_image)[0]
        normalized_prediction = prediction / sum(prediction)
        # normalized_prediction = softmax(prediction)
        return {emotion: normalized_prediction[i] for i, emotion in emotion_dict.items()}

    return ahegao_classification_atulapra


def main():
    # video_name = '6614059835535133954.mp4'
    # video_name = '6620049294030277893.mp4'
    # video_name = '6632512817893215494.mp4'
    # video_name = '6640478247052119301.mp4'
    # video_name = '6642329728030084357.mp4'
    # video_name = '6626761625493835013.mp4'
    # video_name = '6628669426822548741.mp4'
    video_name = '6647420336612576517.mp4'
    # video_name = 'twilight-movie-1--lunch-scene.mp4'

    # todo: probably annotate those few videos and try decision trees/random forest on non-normalized, usually normalized and softmaxed data
    # haar cascade
    face_detection_haar_cascade = init_haar_cascade_detector()

    # ssd neural network from https://github.com/yeephycho/tensorflow-face-detection
    # face_detection_ssd = init_ssd_face_detection()

    # ssd neural network from https://github.com/yeephycho/tensorflow-face-detection
    face_detection_rfcn = init_rfcn_face_detection()

    # ahegao_classifier = init_ahegao_classifier_emopy()
    ahegao_classifier = init_ahegao_classifier_atulapra()
    # visualize_and_eval(video_name, face_detection_haar_cascade)
    # visualize_and_eval(video_name, face_detection_ssd)
    # visualize_and_eval(video_name, face_detection_ssd, ahegao_classifier)
    # visualize_and_eval(video_name, face_detection_rfcn, ahegao_classifier)

    output_file = 'ahegao_analysis_' + video_name
    # visualize_and_eval(video_name, face_detection_rfcn, output_file=output_file)
    visualize_and_eval(video_name, face_detection_rfcn, ahegao_classifier, output_file=output_file)


if __name__ == '__main__':
    main()
