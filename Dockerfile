FROM bvlc/caffe:cpu
WORKDIR /usr/src/app

COPY resnet50_caffe2keras.py .
COPY ResNet-50-deploy.prototxt .
COPY ResNet-50-model.zip .
COPY images /usr/src/app/images
RUN unzip ResNet-50-model.zip

CMD ["python","./resnet50_caffe2keras.py"]