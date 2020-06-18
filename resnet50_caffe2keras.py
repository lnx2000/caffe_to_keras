from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import caffe
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

net = caffe_pb2.NetParameter()

fn = 'ResNet-50-deploy.prototxt'
with open(fn) as f:
    s = f.read()
    txtf.Merge(s, net)

resnet = caffe.Classifier('ResNet-50-deploy.prototxt', 'ResNet-50-model.caffemodel')



def batchw(name1,name2):

  mean = resnet.params[name1][0].data
  variance = resnet.params[name1][1].data

  
  gamma = resnet.params[name2][0].data
  beta = resnet.params[name2][1].data

  return mean,variance,gamma,beta


def convwb(name,bi=False):
  
  kernel=resnet.params[name][0].data
  if bi:
    bias=resnet.params[name][1].data
  else:
    bias=None
  kernel = np.transpose(kernel, (2, 3, 1, 0))


  return kernel,bias

lyr={}
lyr['data']=tf.placeholder(tf.float32,shape=[None,224,224,3])
iterator = iter(range(0, len(net.layer)))
#iterator = iter(range(0, 2))
for i in iterator:
  layer=net.layer[i]
  type=layer.type
  name=layer.name
  if(type=='Convolution'):
    bottom=lyr[layer.bottom[0]]
    top=layer.top[0]
    kernel_size=layer.convolution_param.kernel_size
    pad=layer.convolution_param.pad[0]
    stride=layer.convolution_param.stride
    bi = layer.convolution_param.bias_term

    weights,biases=convwb(name,bi)
    if pad!=0:
      bottom=tf.pad(bottom,tf.constant([[0,0],[pad,pad],[pad,pad],[0,0]]))
    lyr[top]=tf.nn.conv2d(bottom,weights,stride,padding='VALID')

    if bi:
      lyr[top]=lyr[top]+biases
  elif(type=='BatchNorm'):
    bottom=lyr[layer.bottom[0]]
    top=layer.top[0]
    name1=net.layer[i+1].name

    mean,variance,aplha,beta=batchw(name,name1)
    lyr[top]=tf.nn.batch_normalization(bottom,mean,variance,beta,aplha,variance_epsilon=0.001)
    next(iterator,0)
  elif(type=='ReLU'):
    bottom=lyr[layer.bottom[0]]
    top=layer.top[0]
    lyr[top]=tf.nn.relu(bottom)
  elif(type=='Eltwise'):
    bottom1=lyr[layer.bottom[0]]
    bottom2=lyr[layer.bottom[1]]    
    top=layer.top[0]

    lyr[top]=bottom1+bottom2
  elif(type=='Pooling'):
    bottom=lyr[layer.bottom[0]]
    top=layer.top[0]
    kernel=layer.pooling_param.kernel_size
    stride=layer.pooling_param.stride
    typool=layer.pooling_param.pool


    if typool==0:
      bottom=tf.pad(bottom,tf.constant([[0,0],[1,1],[1,1],[0,0]]))
      lyr[top]=tf.nn.max_pool2d(bottom,kernel,stride,padding='VALID')
    else:
      lyr[top]=tf.nn.avg_pool2d(bottom,kernel,stride,padding='VALID')
  elif(type=='InnerProduct'):
    bottom=lyr[layer.bottom[0]]
    top=layer.top[0]
    ip=bottom.shape[-1]
    bottom=tf.reshape(bottom,[-1,ip])

    weights,biases=resnet.params[name][0].data,resnet.params[name][1].data

    weights=np.transpose(weights,(1,0))
    lyr[top]=tf.add(tf.matmul(bottom,weights),biases)

  elif(type=='Softmax'):
    bottom=lyr[layer.bottom[0]]
    lyr['pred']=tf.nn.softmax(bottom)




images=[]
img=cv2.imread('images/jelly_fish.jpg')
images.append(cv2.resize(img,(224,224)))
img=cv2.imread('images/pad_lock.jpg')
images.append(cv2.resize(img,(224,224)))
img=cv2.imread('images/toilet_paper.jpg')
images.append(cv2.resize(img,(224,224)))

images=np.array(images)


sess=tf.Session()
y_pred=sess.run(lyr['pred'],feed_dict={lyr['data']:images})
y_pred=np.argamx(y_pred,1).reshape((3))

cv2.imshow('image1:', images[0])
cv2.imshow('image2:', images[1])
cv2.imshow('image3:', images[2])

print('image1: Actual: %d  Predicted: %d'%(107,y_pred[0]))
print('image2: Actual: %d  Predicted: %d'%(695,y_pred[1]))
print('image3: Actual: %d  Predicted: %d'%(999,y_pred[2]))





