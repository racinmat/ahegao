from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import cv2
import progressbar

from ahegao_detection_eval_show_video import init_haar_cascade_detector, init_ssd_face_detection, VIDEOS_DIR, \
    init_rfcn_face_detection


def eval_video(detector, video_name):
    detector_name = detector.__name__
    print('processing: ', video_name, ' by: ', detector_name)

    video_path = osp.join(VIDEOS_DIR, video_name)
    if not osp.exists(video_path):
        raise Exception(f'Video {video_path} not found.')

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    widgets = [progressbar.Percentage(), ' ', progressbar.Counter(), ' ', progressbar.Bar(), ' ',
               progressbar.FileTransferSpeed()]
    pbar = progressbar.ProgressBar(widgets=widgets, max_value=length).start()

    i = 0
    data_x = []
    data_y = []
    while True:
        pbar.update(i)
        ret, frame = cap.read()
        if not ret:
            break

        faces, scores = detector(frame)
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

        # update x and ylim to show all points:
        plt.xlim(min(data_x) - 0.5, max(data_x) + 0.5)
        plt.ylim(min(data_y) - 0.5, max(data_y) + 0.5)

        i += 1

    pbar.finish()
    cap.release()

    plt.figure()
    plt.plot(data_x, data_y)
    plt.savefig(f'plots/{video_name}_{detector_name}.png')

    face_detections = np.array(data_y)
    with open(f'plots/{video_name}_{detector_name}.txt', 'w') as f:
        with redirect_stdout(f):
            print('face mean: ', face_detections.mean())
            print('face variance: ', face_detections.var())
            print('face quantiles: ', np.percentile(face_detections, [1, 25, 50, 75, 99]))


def main():
    videos = [
        '6620049294030277893.mp4',
        '6628669426822548741.mp4',
        '6614059835535133954.mp4',
        '6647420336612576517.mp4',
        'twilight-movie-1--lunch-scene.mp4'
    ]

    # haar cascade
    face_detection_haar_cascade = init_haar_cascade_detector()

    # ssd neural network from https://github.com/yeephycho/tensorflow-face-detection
    face_detection_ssd = init_ssd_face_detection()

    face_detection_rfcn = init_rfcn_face_detection()

    for video_name in videos:
        eval_video(face_detection_haar_cascade, video_name)
        eval_video(face_detection_ssd, video_name)
        eval_video(face_detection_rfcn, video_name)


if __name__ == '__main__':
    main()
