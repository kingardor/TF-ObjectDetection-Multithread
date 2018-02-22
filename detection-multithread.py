'''  Python Script for Oculus

Usage: python3 detection-multithread.py
Flag list: --width - to set Width
           --height - to set Height
           --source - to set source of input

Press 'q' to exit
'''
import os
import argparse

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, draw_boxes_and_labels
# from utils.app_utils import WebcamVideoStream  # for WebcamVideoStream
from object_detection.utils import label_map_util

import numpy as np
import cv2
import tensorflow as tf

# to obtain curent working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(
    CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def argsparser():
    ''' argsparser() adds argument functionality for command line. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', dest='source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    return parser.parse_args()


def detect_objects(image_np, sess, detection_graph):
    '''detect_objects is used to find objects in the frame and draw a box'''
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def detect(input_q, output_q):
    '''Isolated function to enable threading of object detection'''
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    while True:
        fram = input_q.get()
        frame_rgb = cv2.cvtColor(fram, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    sess.close()


def edge(input_q, output_q):
    '''Isolated function to enable threading of edge detection'''
    while True:
        image = input_q.get()
        # load the image, convert it to grayscale, and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # apply Canny edge detection using a wide threshold, tight
        # threshold, and automatically determined threshold
        wide = cv2.Canny(blurred, 10, 200)
        output_q.put(wide)


if __name__ == '__main__':
    args = argsparser()

    # Queueing is used to dramaticallty improve framerates
    # Queue for Object Detection

    # Queue for Edge Detection
    ed_input_q = Queue(5)
    # fps is better if queue is higher but then more lags
    ed_output_q = Queue()
    for i in range(1):
        ed_t = Thread(target=edge, args=(ed_input_q, ed_output_q))
        ed_t.daemon = True
        ed_t.start()

    ob_input_q = Queue(5)
    # fps is better if queue is higher but then more lags
    ob_output_q = Queue()
    for i in range(1):
        ob_t = Thread(target=detect, args=(ob_input_q, ob_output_q))
        ob_t.daemon = True
        ob_t.start()

    # video_capture = WebcamVideoStream(src=args.video_source,
    #                                  width=args.width,
    #                                  height=args.height).start()

    video_capture = cv2.VideoCapture(args.source)
    fps = FPS().start()

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)  # to flip image on coorect orientation
        frame = cv2.resize(frame, (args.width, args.height))
        # print(type(frame))
        ed_input_q.put(frame)
        ob_input_q.put(frame)

        #ob_t = time.time()

        if ed_output_q.empty():
            pass  # fill up Queue
        else:
            img = ed_output_q.get()
            cv2.imshow('Edge', img)

        if ob_output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = ob_output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                cv2.rectangle(frame, (int(point['xmin'] * args.width),
                                      int(point['ymin'] * args.height)),
                              (int(point['xmax'] * args.width),
                               int(point['ymax'] * args.height)), color, 3)
                cv2.rectangle(frame, (int(point['xmin'] * args.width),
                                      int(point['ymin'] * args.height)),
                              (int(point['xmin'] * args.width) + len(name[0]) * 6,
                               int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[0], (int(point['xmin'] * args.width),
                                             int(point['ymin'] * args.height)),
                            font, 0.3, (0, 0, 0), 1)
            cv2.imshow('Object', frame)

        fps.update()

        # Exit if input key is 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    # video_capture.stop() # if using WebcamVideoStream
    video_capture.release()  # if using cv2.VideoCapture(0)
    cv2.destroyAllWindows()
