FROM bvlc/caffe:cpu
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless

COPY resnet50_caffe2keras.py .
COPY ResNet-50-deploy.prototxt .
COPY ResNet-50-model.caffemodel .
COPY images /usr/src/app/images


CMD ["python","./resnet50_caffe2keras.py"]