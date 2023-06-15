from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import cv2
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def img_prediction(batch):
    prediction = model(batch)[0]
    print(prediction)
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                              labels=labels,
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
    output_video = cv2.VideoWriter('output_video.mp4',
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (frame_width, frame_height))
    # Loop over the frames
    while video.isOpened():
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
        print(prediction)
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=4, font_size=30)
        im = to_pil_image(box.detach())
        # im.show()

        opencv_image = np.array(im)
        opencv_image = cv2.convertScaleAbs(opencv_image)
        height, width, channels = opencv_image.shape

        output_video.write(opencv_image)

    output_video.release()



if __name__ == "__main__":
    img = read_image("soccer.jpg")
    video = cv2.VideoCapture('ayaka.mp4')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device {}".format(device))

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.to(device)
    model.eval()

    print(model)

    preprocess = weights.transforms()
    batch = [preprocess(img).to(device)]


    # img_prediction(batch)
    # video_prediction(video)
