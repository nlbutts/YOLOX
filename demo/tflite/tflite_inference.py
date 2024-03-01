import numpy as np
import tensorflow as tf
import cv2
from demo_utils import *
from visualize import *
import os
from timeit import TimeIt

COCO_CLASSES = ['seeds']

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolox_canola.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data. Replace this with your input data.
input_shape = input_details[0]['shape']

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img = np.expand_dims(padded_img, axis=0)
    return padded_img, r

infile = '/home/nlbutts/projects/YOLOX/Canola/Run_3/BaslerLogger/5472.bmp'
org_img = cv2.imread(infile)
img, ratio = preproc(org_img, np.array((input_shape[2], input_shape[3])))

runs = 100
t = TimeIt('Infer: ')
for i in range(runs):
    # Set input tensor.
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference.
    interpreter.invoke()

    # Get output tensor.
    output = interpreter.get_tensor(output_details[0]['index'])

t.stop()
t.print()
print("Output shape:", output.shape)
print("Output result:", output)

input_shape = (input_shape[2], input_shape[3])
predictions = demo_postprocess(output, input_shape)[0]

boxes = predictions[:, :4]
scores = predictions[:, 4:5] * predictions[:, 5:]

boxes_xyxy = np.ones_like(boxes)
boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
boxes_xyxy /= ratio
dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
if dets is not None:
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    origin_img = vis(org_img, final_boxes, final_scores, final_cls_inds,
                        conf=0.4, class_names=COCO_CLASSES)

output_path = os.path.basename('5472_out.bmp')
cv2.imwrite(output_path, origin_img)
