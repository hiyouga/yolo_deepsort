import logging
import numpy as np
from deep_sort import DeepSort
from yolo3.detect.video_detect import VideoDetector
from yolo3.models import Darknet

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    model = Darknet("config/yolov3.cfg", img_size=(608, 608))
    model.load_darknet_weights("weights/yolov3.weights")
    model.to("cuda:0")

    tracker = DeepSort("weights/ckpt.t7",
                       min_confidence=1,
                       use_cuda=True,
                       nn_budget=30,
                       n_init=3,
                       max_iou_distance=0.7,
                       max_dist=0.3,
                       max_age=30)

    video_detector = VideoDetector(model, "config/coco.names",
                                   thickness=2,
                                   skip_frames=-1,
                                   thres=0.5,
                                   class_mask=[0],
                                   nms_thres=0.4,
                                   tracker=tracker,
                                   half=True)

    filename = 'data/a.csv'
    counts = list()
    for i, (image, detections, video_time, real_time) in enumerate(video_detector.detect('data/a.mp4',
                                                                                         output_path='data/a.ts',
                                                                                         real_show=False,
                                                                                         skip_secs=0)):
        # cv2.imwrite('data/frames/{}.jpg'.format(i), image)
        counts.append(len(detections))
    np.savetxt(filename, counts, fmt='%d', encoding='utf-8')
