import cv2
import numpy as np
import sys
import multiprocessing as mp
import time

def init_net():
    CONFIG = 'yolov4-tiny-myobj.cfg'
    WEIGHT = 'yolov4-tiny-myobj_2000.weights'

    net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHT)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    dummy = np.zeros((512, 512, 3), np.uint8)
    dummy.fill(0)
    blob = cv2.dnn.blobFromImage(dummy, 1/255.0, (416, 416))
    net.setInput(blob)
    net.forward(output_layers)

    return net, output_layers

def object_detect(frame_queue, object_queue):
    net, output_layers = init_net()
    while True:
        image = frame_queue.get()
        blob = cv2.dnn.blobFromImage(
            image,
            1/255.0,     # 圖片正規化
            (416, 416),  # 根據 cfg 檔
            (0, 0, 0),   # 均值設定
            True,        # 圖片通道為BGR時設定為 True
            crop=False)  # 不剪裁圖片
        net.setInput(blob)
        outs = net.forward(output_layers)
        object_queue.put((image, outs))

def object_info(object_queue, output_queue):
    NAMES = '/home/eddie/train/obj.names'
    colors = []
    classes = []
    with open(NAMES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

    while True:
        class_ids = []
        confidences = []
        boxes = []

        (image, outs) = object_queue.get()
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    x, y, w, h = get_box(image, detection)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        image = draw_box(image, class_ids, confidences, boxes, classes, colors)
        output_queue.put(image)

def get_box(image, detection):
    height, width, channels = image.shape
    center_x = int(detection[0] * width)
    center_y = int(detection[1] * height)
    w = int(detection[2] * width)
    h = int(detection[3] * height)
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)
    return x, y, w, h

def draw_box(image, class_ids, confidences, boxes, classes, colors):
    image_new = image.copy()
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[classes.index(label)]
            label = '{}:{:.2f}'.format(label, confidences[i])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(image_new, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image_new, label, (x, y - 10), font, 0.7, color, 2)

    return image_new

def capture_video(frame_queue):
    IMAGE = sys.argv[1]
    if IMAGE != 0:
        cap = cv2.imread(sys.argv[1])
    if IMAGE == '0':
        IMAGE = 0
        cap = cv2.VideoCapture(IMAGE, cv2.CAP_V4L2)
    
    cap.set(3, 640)
    cap.set(4, 480)
    ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    WIDTH = 480
    HEIGHT = int(WIDTH / ratio)

    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (WIDTH, HEIGHT))
        frame_queue.put(image)


def main():
    pool = mp.Pool()
    frame_queue = mp.Manager().Queue(1)
    object_queue = mp.Manager().Queue(1)
    output_queue = mp.Manager().Queue(1)

    pool.apply_async(capture_video, [frame_queue])
    pool.apply_async(object_detect, [frame_queue, object_queue])
    pool.apply_async(object_info, [object_queue, output_queue])
    
    #time.sleep(2)
   # fps = [0]*60 #count average every 30 frames
    fps = []
    while True:
        begin_time = time.time()
        frame = output_queue.get()
       
        fps.append(1/(time.time() - begin_time))
        print(time.time()-begin_time)
        #fps = fps[1:]
        text = 'FPS:{:.2f}'.format(sum(fps)/len(fps))
        cv2.putText(
                frame, text, (10,30),
                cv2.FONT_HERSHEY_PLAIN, 1.4, (0,255,255), 1
        )
        cv2.imshow('video', frame)        
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break

main()
    


