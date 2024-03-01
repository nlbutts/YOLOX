import onnx
from onnx_tf.backend import prepare
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(
                    prog='onnx2tflite',
                    description='Reads the onnx model and converts to TFLite')
parser.add_argument('-i', '--onnx', type=str, required=True, help='Path to the onnx model')
parser.add_argument('-t', '--tfdir', type=str, required=True, help='Tensorflow output directory')
parser.add_argument('-f', '--tflite', type=str, required=True, help='TFLite model output name')

args = parser.parse_args()


onnx_model = onnx.load(args.onnx)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(args.tfdir)
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(args.tfdir)
tflite_model = converter.convert()

# Save the model
with open(args.tflite, 'wb') as f:
    f.write(tflite_model)