from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import caffe
#import numpy as np
#import cv2
#import tensorflow.compat.v1 as tf

#tf.disable_eager_execution()

net = caffe_pb2.NetParameter()
fn = 'ResNet-50-deploy.prototxt'
with open(fn) as f:
    s = f.read()
    txtf.Merge(s, net)
resnet = caffe.Classifier('ResNet-50-deploy.prototxt', 'ResNet-50-model.caffemodel')
