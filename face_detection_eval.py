from EmoPy.src.fermodel import FERModel
from docutils.nodes import thead
from pkg_resources import resource_filename
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_and_eval(cap, line, detector):
    i = 0
    data_x = []
    data_y = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector(frame)
        # compute softmax probabilities
        if len(faces) > 0:
            # draw box around face with maximum area
            max_area_face = faces[0]
            for face in faces:
                # multi face setup
                (x, y, w, h) = face
                frame = cv2.rectangle(frame, (x, y - 30), (x + w, y + h + 10), (255, 0, 0), 2)

                # single face setup
                # if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                #     max_area_face = face
            # face = max_area_face
            # (x, y, w, h) = max_area_face
            # frame = cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

        data_x.append(i)
        data_y.append(len(faces))
        line.set_xdata(data_x)
        line.set_ydata(data_y)
        plt.draw()
        plt.pause(0.0001)

        # update x and ylim to show all points:
        plt.xlim(min(data_x) - 0.5, max(data_x) + 0.5)
        plt.ylim(min(data_y) - 0.5, max(data_y) + 0.5)

        cv2.imshow('Video', cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def init_ssd_face_detection():
    PATH_TO_CKPT = 'tensorflow_face_detection/model/frozen_inference_graph_face.pb'
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
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
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        return sess, boxes, image_tensor, scores


def main():
    cv2.ocl.setUseOpenCL(False)
    plt.ion()
    line, = plt.plot([], [], 'r-')

    # cap = cv2.VideoCapture('videos/6614059835535133954.mp4')
    # cap = cv2.VideoCapture('videos/6620049294030277893.mp4')
    # cap = cv2.VideoCapture('videos/6628669426822548741.mp4')
    cap = cv2.VideoCapture('videos/6647420336612576517.mp4')

    # haar cascade
    facecasc = cv2.CascadeClassifier('Emotion/models/haarcascade_frontalface_default.xml')

    def face_detection_haar_cascade(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        # faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
        return faces

    # ssd neural network from https://github.com/yeephycho/tensorflow-face-detection
    sess, boxes_tensor, image_tensor, scores_tensor = init_ssd_face_detection()

    def face_detection_ssd(frame):
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        boxes, scores = sess.run([boxes_tensor, scores_tensor], feed_dict={image_tensor: image_np_expanded})
        # to format returned by opencv
        threshold = 0.5
        boxes = boxes[scores > threshold, :]
        # ymin, xmin, ymax, xmax = box
        faces = np.zeros_like(boxes)
        faces[:, 1] = boxes[:, 0] * frame.shape[0]
        faces[:, 0] = boxes[:, 1] * frame.shape[1]
        faces[:, 3] = (boxes[:, 2] - boxes[:, 0]) * frame.shape[0]
        faces[:, 2] = (boxes[:, 3] - boxes[:, 1]) * frame.shape[1]
        faces = faces.astype(np.int32)
        return faces

    # visualize_and_eval(cap, line, face_detection_haar_cascade)
    visualize_and_eval(cap, line, face_detection_ssd)


if __name__ == '__main__':
    main()
