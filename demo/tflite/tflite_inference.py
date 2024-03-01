import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import argparse
from pathlib import Path
import time

from yolox.utils import demo_postprocess, vis, multiclass_nms

class TFLiteDetect():
    def __init__(self, model_path, outdir, tflite_threads=4):
        # Load TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path=model_path, num_threads=tflite_threads)
        self.interpreter.allocate_tensors()

        self.outdir = Path(outdir)

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(self.input_details)
        print(self.output_details)

        # Prepare input data. Replace this with your input data.
        self.input_shape = self.input_details[0]['shape']
        self.input_size = [self.input_shape[2], self.input_shape[3]]
        self.quantization = self.output_details[0]['quantization']
        self.input_dt = self.input_details[0]['dtype']
        self.output_dt = self.output_details[0]['dtype']

    def preproc(self, img, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114

        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=self.input_dt)
        padded_img = np.expand_dims(padded_img, axis=0)
        return padded_img, r

    def detect(self, org_img, outpath):
        img, ratio = self.preproc(org_img)

        # Set input tensor.
        self.interpreter.set_tensor(self.input_details[0]['index'], img)

        # Run inference.
        self.interpreter.invoke()

        # Get output tensor.
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        if self.quantization:
            output = output.astype('float32')
            output = (output - self.quantization[1]) * self.quantization[0]

        predictions = demo_postprocess(output, self.input_size)[0]

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

    def process_file(self, file):
        img = cv2.imread(str(file))
        outfile = outdir / file.name
        self.detect(img, str(outfile))

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

outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True)
indata = Path(args.input)

tfdet = TFLiteDetect(args.tflite, args.outdir, args.threads)

if indata.is_dir():
    file_types = ['*.jpg', '*.bmp', '*.png']
    files = [file for file_type in file_types for file in indata.glob(file_type)]
    start = time.time()
    for file in files:
        print(f'Processing {file}')
        tfdet.process_file(file)
    stop = time.time()
    diff = stop - start
    time_per_image = diff / len(files)
    print(f'Total time: {diff} Time/image: {time_per_image}  FPS: {1/time_per_image}')
else:
    img = cv2.imread(infile)
    tfdet.process_File(args.input)

