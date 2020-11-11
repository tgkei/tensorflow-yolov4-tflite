from pprint import pprint
import json
import tensorflow as tf
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import cv2
import numpy as np

MODEL_PATH = './checkpoints/yolov4-416'
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

# load model
saved_model_loaded = tf.saved_model.load(
    MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


def main(video_path):
    ret_dict = dict()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img_input = img_input / 255.
        img_input = img_input[np.newaxis, ...].astype(np.float32)
        img_input = tf.constant(img_input)

        pred_bbox = infer(img_input)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=SCORE_THRESHOLD
        )

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(),
                     valid_detections.numpy()]

        detected_nums = pred_bbox[3][0]
        #result = utils.draw_bbox(img, pred_bbox)

        class_name = utils.read_class_names(cfg.YOLO.CLASSES)

        num_classes = len(class_name)
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox

        if detected_nums:
            frame = cap.get(cv2.CAP_PROP_POS_MSEC)
            #print("Detected at: "+str(cap.get(cv2.CAP_PROP_POS_MSEC)))

        for i in range(num_boxes[0]):
            class_ind = int(out_classes[0][i])
            if class_name[class_ind] not in ret_dict:
                ret_dict[class_name[class_ind]] = set()
            ret_dict[class_name[class_ind]].add(frame)

        #result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        #cv2.imshow('result', result)
        if cv2.waitKey(1) == ord('q'):
            break

    return ret_dict


if __name__ == '__main__':
    #video_path = './data/road.mp4'
    video_path = './data/blur_output_H264_573.mp4'
    time_info = main(video_path)
    tmp = dict()
    for class_name, times in time_info.items():
        times = sorted(list(times))
        tmp[class_name] = times
    time_info = json.dumps(tmp)
    with open("test.json", "w") as json_file:
        json.dump(time_info, json_file)
