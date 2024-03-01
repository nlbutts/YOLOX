import onnx
from onnx_tf.backend import prepare
import argparse
import tensorflow as tf
import cv2
import numpy as np
from pathlib import Path

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

parser = argparse.ArgumentParser(
                    prog='onnx2tflite',
                    description='Reads the onnx model and converts to TFLite')
parser.add_argument('-i', '--onnx', type=str, required=True, help='Path to the onnx model')
parser.add_argument('-t', '--tfdir', type=str, required=True, help='Tensorflow output directory')
parser.add_argument('-f', '--tflite', type=str, required=True, help='TFLite model output name')
parser.add_argument('-d', '--dataset', type=str, required=False, help='Directory of files for the representative data set')
parser.add_argument('--int8', action='store_true', required=False, help='Quantitize the model')

args = parser.parse_args()

onnx_model = onnx.load(args.onnx)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(args.tfdir)
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(args.tfdir)

if args.int8:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_gen():
        file_types = ['*.jpg', '*.bmp', '*.png']
        indata = Path(args.dataset)
        if not indata.is_dir():
            print("This is not a directory, please select a directory")
            exit(0)
        files = [file for file_type in file_types for file in indata.glob(file_type)]
        sub_files = np.random.choice(files, 100, replace=False)
        for file in sub_files:
            orgimg = cv2.imread(str(file))
            img, ratio = preproc(orgimg, (320, 320))
            # get sample input data as numpy array
            yield [img]

    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save the model
with open(args.tflite, 'wb') as f:
    f.write(tflite_model)