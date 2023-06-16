from torchvision.io.image import read_image
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import cv2
from PIL import Image
import torch
import numpy as np
import time
from utils import coco_names
from utils import pil2cv


def img_prediction(batch):
    prediction = model(batch)[0]
    boxes, scores, labels = prediction["boxes"], prediction["scores"], prediction["labels"]

    num = torch.argwhere(scores > 0.5).shape[0]
    boxes = boxes[0:num]
    class_names = []

    for i in range(num):
        class_names.append(coco_names[labels[i]-1])

    box = draw_bounding_boxes(img, boxes=boxes,
                              labels=class_names,
                              colors="red",
                              width=4, font_size=30)
    im = to_pil_image(box.detach())
    im.show()


def video_prediction(video):
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the output video
    output_video = cv2.VideoWriter('video/output_video.mp4',
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (frame_width, frame_height))
    # Loop over the frames
    time1 = time.time()
    while video.isOpened():
        # if time.time()-time1 > 90:
        #     break
        # Read the next frame
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
        frame = Image.fromarray(frame)
        processed_frame = preprocess(frame)
        processed_frame = processed_frame.to(device)
        batch = [processed_frame]


        # Run object detection
        prediction = model(batch)[0]
        # print(prediction)
        boxes, scores, labels = prediction["boxes"], prediction["scores"], prediction["labels"]

        num = torch.argwhere(scores > 0.5).shape[0]
        boxes = boxes[0:num]
        class_names = []

        for i in range(num):
            class_names.append(coco_names[labels[i] - 1])

        box = draw_bounding_boxes(img, boxes=boxes,
                                  labels=class_names,
                                  colors="red",
                                  width=4, font_size=30)
        im = to_pil_image(box.detach())
        # im.show()

        opencv_image = np.array(im)
        opencv_image = cv2.convertScaleAbs(opencv_image)
        height, width, channels = opencv_image.shape

        output_video.write(opencv_image)

    output_video.release()

def camera_prediction():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()  # 读取一帧图像
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
        frame = Image.fromarray(frame)
        processed_frame = preprocess(frame)
        processed_frame = processed_frame.to(device)
        batch = [processed_frame]

        # Run object detection
        prediction = model(batch)[0]
        # print(prediction)
        boxes, scores, labels = prediction["boxes"], prediction["scores"], prediction["labels"]

        num = torch.argwhere(scores > 0.5).shape[0]
        boxes = boxes[0:num]
        class_names = []

        for i in range(num):
            class_names.append(coco_names[labels[i] - 1])

        box = draw_bounding_boxes(img, boxes=boxes,
                                  labels=class_names,
                                  colors="red",
                                  width=4, font_size=30)
        im = to_pil_image(box.detach())
        predict_frame = pil2cv(im)
        cv2.imshow("Live", predict_frame)
        if cv2.waitKey(1) == ord('q'):  # 按下 'q' 键退出循环
            break
    cap.release()
    cv2.destroyAllWindows()






if __name__ == "__main__":
    start_time = time.time()

    img = read_image("img/soccer.jpg")
    video = cv2.VideoCapture('video/barbara.mp4')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device {}".format(device))

    # Step 1: Initialize model with the best available weights
    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights, box_score_thresh=0.9)
    model.to(device)
    model.eval()


    preprocess = weights.transforms()
    batch = [preprocess(img).to(device)]


    # img_prediction(batch)
    video_prediction(video)
    # camera_prediction()

    end_time = time.time()

    # 计算代码的运行时间
    execution_time = end_time - start_time
    print("execution time:{}".format(execution_time))

