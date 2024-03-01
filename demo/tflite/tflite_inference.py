import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import argparse
from pathlib import Path
import time

from yolox.utils import demo_postprocess, vis, multiclass_nms

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

def detect(interpreter, org_img, input_shape, outpath):
    img, ratio = preproc(org_img, np.array((input_shape[2], input_shape[3])))

    # Set input tensor.
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference.
    interpreter.invoke()

    # Get output tensor.
    output = interpreter.get_tensor(output_details[0]['index'])

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
    origin_img = None
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(org_img, final_boxes, final_scores, final_cls_inds,
                            conf=0.4, class_names=COCO_CLASSES)

        cv2.imwrite(outpath, origin_img)

parser = argparse.ArgumentParser(
                    prog='tflite_infer',
                    description='Performs tflite inference')
parser.add_argument('-f', '--tflite', type=str, required=True, help='TFLite model file')
parser.add_argument('-i', '--input', type=str, required=True, help='Input file')
parser.add_argument('-o', '--outdir', type=str, required=True, help='Output directory for the inference')
parser.add_argument('-t', '--threads', type=int, required=False, default=4, help='Number of threads to use')

args = parser.parse_args()

COCO_CLASSES = ['seeds']

infile = args.input

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=args.tflite, num_threads=args.threads)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data. Replace this with your input data.
input_shape = input_details[0]['shape']

outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True)
indata = Path(args.input)
if indata.is_dir():
    file_types = ['*.jpg', '*.bmp', '*.png']
    files = [file for file_type in file_types for file in indata.glob(file_type)]
    start = time.time()
    for file in files:
        print(f'Processing {file}')
        outfile = outdir / file.name
        img = cv2.imread(str(file))
        detect(interpreter, img, input_shape, str(outfile))

    stop = time.time()
    diff = stop - start
    time_per_image = diff / len(files)
    print(f'Total time: {diff} Time/image: {time_per_image}  FPS: {1/time_per_image}')
else:
    img = cv2.imread(infile)
    detect(interpreter, img, input_shape, args.outdir)

