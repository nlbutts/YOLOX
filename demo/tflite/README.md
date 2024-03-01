## TFLite in Python

This doc introduces how to convert your onnx model to tflite and run inference.

### Step1: Install onnxruntime

run the following command to install onnxruntime:
```shell
pip install tflite-runtime tensorflow tensorflow_probability onnx onnx_tf opencv-python
```

I normally setup a virtualenv, install the YOLOX dependencies, then the above dependencies.

### Step2: Convert

Convert your model to TFLite.

'''
python onnx2tflite.py --onnx model.onnx -t modelpb -f model.tflite
'''

### Step3: Run inference


'''
python tflite_inference.py -f model.tflite -i <input file directory> -o <output directory> -t <threads>
'''