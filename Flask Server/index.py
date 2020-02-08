import pyrebase
from flask import Flask, request, render_template, jsonify
import cv2
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Input , Conv2D , MaxPooling2D , UpSampling2D , Dropout , Activation
from tensorflow.keras.optimizers import Adam , SGD
from tensorflow.keras.layers import LeakyReLU as LeakyRelu
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add , Concatenate , Reshape
from tensorflow.keras.callbacks import  ModelCheckpoint , ReduceLROnPlateau , EarlyStopping ,TensorBoard
from tensorflow.keras.initializers import RandomNormal


config = {
	"apiKey": "AIzaSyAlKzu4OhAgcvPr5TJXBtvENUUXzp2sXec",
    "authDomain": "mango-prediction-1691d.firebaseapp.com",
    "databaseURL": "https://mango-prediction-1691d.firebaseio.com",
    "projectId": "mango-prediction-1691d",
    "storageBucket": "mango-prediction-1691d.appspot.com",
    "messagingSenderId": "1073248007466",
    "appId": "1:1073248007466:web:c9605f30a433cd22739a53"
}

app = Flask(__name__)

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()



# THe Neural Network model for our Predict Model  ( Mango Net )
def initialize():
  #return RandomNormal()
  return 'he_normal'

def unet(drop=0.2):
  inputs = Input(shape=(200,200,3))

  ##Encoding Blocks

  #block1
  block1 = Conv2D(filters=16,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(inputs)
  block1 = BatchNormalization()(block1)
  block1 = Activation('relu')(block1)
  block1 = Conv2D(filters=16,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block1)

  batch1 = Conv2D(filters=16,kernel_size=(1,1),padding='same',strides=1,kernel_initializer=initialize())(inputs)
  batch1 = BatchNormalization()(batch1)

  block1 = Add()([block1,batch1])

  #block2
  block2 = BatchNormalization()(block1)
  block2 = Activation('relu')(block2)
  block2 = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=2,kernel_initializer=initialize())(block2)
  block2 = BatchNormalization()(block2)
  block2 = Activation('relu')(block2)
  block2 = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block2)

  batch2 = Conv2D(filters=32,kernel_size=(1,1),padding='same',strides=2,kernel_initializer=initialize())(block1)
  batch2 = BatchNormalization()(batch2)

  block2 = Add()([block2,batch2])

  #block3
  block3 = BatchNormalization()(block2)
  block3 = Activation('relu')(block3)
  block3 = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=2,kernel_initializer=initialize())(block3)
  block3 = BatchNormalization()(block3)
  block3 = Activation('relu')(block3)
  block3 = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block3)

  batch3 = Conv2D(filters=64,kernel_size=(1,1),padding='same',strides=2,kernel_initializer=initialize())(block2)
  batch3 = BatchNormalization()(batch3)

  block3 = Add()([block3,batch3])

  #block4
  block4 = BatchNormalization()(block3)
  block4 = Activation('relu')(block4)
  block4 = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=2,kernel_initializer=initialize())(block4)
  block4 = BatchNormalization()(block4)
  block4 = Activation('relu')(block4)
  block4 = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block4)

  batch4 = Conv2D(filters=128,kernel_size=(1,1),padding='same',strides=2,kernel_initializer=initialize())(block3)
  batch4 = BatchNormalization()(batch4)

  block4 = Add()([block4,batch4])


  ##Bridge Building

  #Bridge block1
  bridge1 = BatchNormalization()(block4)
  bridge1 = Activation('relu')(bridge1)
  bridge1 = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(bridge1)
  bridge1 = BatchNormalization()(bridge1)
  bridge1 = Activation('relu')(bridge1)
  bridge1 = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(bridge1)

  bridge_bactch1 = Conv2D(filters=256,kernel_size=(1,1),padding='same',strides=1,kernel_initializer=initialize())(block4)
  bridge_bactch1 = BatchNormalization()(bridge_bactch1)

  bridge1 = Add()([bridge1,bridge_bactch1])

  #Bridge block2
  bridge2 = BatchNormalization()(bridge1)
  bridge2 = Activation('relu')(bridge2)
  bridge2 = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(bridge2)
  bridge2 = BatchNormalization()(bridge2)
  bridge2 = Activation('relu')(bridge2)
  bridge2 = Conv2D(filters=256,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(bridge2)

  bridge_bactch2 = Conv2D(filters=256,kernel_size=(1,1),padding='same',strides=1,kernel_initializer=initialize())(bridge1)
  bridge_bactch2 = BatchNormalization()(bridge_bactch2)

  bridge2 = Add()([bridge2,bridge_bactch2])

  ##Decoding Blocks

  #block5
  block5_1 = UpSampling2D((2,2))(bridge2)
  block5_1 = Concatenate()([block5_1 , block3])


  block5 = BatchNormalization()(block5_1)
  block5 = Activation('relu')(block5)
  block5 = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block5)
  block5 = BatchNormalization()(block5)
  block5 = Activation('relu')(block5)
  block5 = Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block5)

  batch5 = Conv2D(filters=128,kernel_size=(1,1),padding='same',strides=1,kernel_initializer=initialize())(block5_1)
  batch5 = BatchNormalization()(batch5)

  block5 = Add()([block5,batch5])
  block5 = Dropout(drop)(block5)

  #block6
  block6_1 = UpSampling2D((2,2))(block5)
  block6_1 = Concatenate()([block6_1 , block2])

  block6 = BatchNormalization()(block6_1)
  block6 = Activation('relu')(block6)
  block6 = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block6)
  block6 = BatchNormalization()(block6)
  block6 = Activation('relu')(block6)
  block6 = Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block6)

  batch6 = Conv2D(filters=64,kernel_size=(1,1),padding='same',strides=1,kernel_initializer=initialize())(block6_1)
  batch6 = BatchNormalization()(batch6)

  block6 = Add()([block6,batch6])
  block6 = Dropout(drop)(block6)

  #block7
  block7_1 = UpSampling2D((2,2))(block6)
  block7_1 = Concatenate()([block7_1 , block1])

  block7 = BatchNormalization()(block7_1)
  block7 = Activation('relu')(block7)
  block7 = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block7)
  block7 = BatchNormalization()(block7)
  block7 = Activation('relu')(block7)
  block7 = Conv2D(filters=32,kernel_size=(3,3),padding='same',strides=1,kernel_initializer=initialize())(block7)

  batch7 = Conv2D(filters=32,kernel_size=(1,1),padding='same',strides=1,kernel_initializer=initialize())(block7_1)
  batch7 = BatchNormalization()(batch7)

  block7 = Add()([block7,batch7])
  block7 = Dropout(drop)(block7)


  #final block
  outputs = Conv2D(filters=1,kernel_size=(1,1),strides=1,padding='same',kernel_initializer=initialize())(block7)
  outputs = Reshape((200,200))(outputs)
  outputs = Activation('sigmoid')(outputs)

  model = Model(inputs,outputs)

  return model





# utility function needed for processing the image
def pathed_image (image):
  patchs=[]
  h=15
  w=20
  for i in range(1,h+1):
      for j in range(1,w+1):
          img = image[ 200*i-200: 200 *i , 200*j-200 : 200*j]
          patchs.append(img)
  return np.array(patchs)

def predict(model,image):
  patch = pathed_image(image)
  h1=[]
  h=15
  pred = model.predict(patch)


  for i in (range(1,h+1)):
    img1 = np.hstack(pred[20*i-20 : 20*i])
    h1.append(img1)
    
  fimg = np.vstack(h1)
  #plt.imshow(fimg,cmap='gray')
  return fimg



def dice_coef(y_true, y_pred):
	smooth = 1.0
	y_true_f = tf.layers.flatten(y_true)
	y_pred_f = tf.layers.flatten(y_pred)
	intersection = tf.reduce_sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return 1.0 - dice_coef(y_true, y_pred)

@app.route('/')
def demo():
	app.logger.info("****************************   Started the Process  *******************************")
	print("****************************   Started the Process  *******************************")
	storage.child('input.jpg').download('input.jpg')

	app.logger.info("****************************   Image Downloaded Successfully  *******************************")
	print("****************************   Image Downloaded Successfully  *******************************")
	img = cv2.imread('input.jpg')
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	if img.shape[0]!=4000 and img.shape[1]!=3000:
		img = cv2.resize(img, (4000,3000), interpolation = cv2.INTER_AREA)
	model = unet()
	model.compile(optimizer='adam',loss=dice_coef_loss,metrics=['acc',keras.metrics.Precision(),'FalseNegatives','FalsePositives','TrueNegatives','TruePositives','CosineSimilarity','AUC'])
	model.load_weights("unet.h5")
	app.logger.info("****************************   Loaded the Model Successfully  *******************************")
	print("****************************   Loaded the Model Successfully  *******************************")
	out = predict(model,img)
	app.logger.info("****************************   Predictions Made  *******************************")
	print("****************************   Predictions Made  *******************************")
	plt.imsave("preds.jpeg",out)
	outs = cv2.imread("preds.jpeg",0)
	outs = cv2.threshold(outs,175,255,cv2.THRESH_BINARY)[1]
	con, hier = cv2.findContours(outs , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(img,con,-1,(255,0,0),10)
	for c in con:
		x,y,w,h = cv2.boundingRect(c)
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),10)

	app.logger.info("****************************   Bounding Boxes are Placed  *******************************")
	print("****************************   Bounding Boxes are Placed  *******************************")
	plt.imsave('output.jpeg',img)

	app.logger.info("****************************   Saving the Output Image  *******************************")
	print("****************************   Saving the Output Image  *******************************")
	storage.child("output.jpeg").put("output.jpeg")
	#storage.child("preds.jpeg").put("preds.jpeg")
	app.logger.info("****************************   All Process Completed  *******************************")
	print("****************************   All Process Completed  *******************************")
	output = {}
	output['count'] = str(len(con))
	output['url'] = str(storage.child("output.jpeg").get_url(None))
	print(output)


	return jsonify(output)


@app.route('/checkapi')
def check():

  d={}
  d['check']='demo'
  d['hello']=69
  d["api"]='flask'

  return jsonify(d)

if __name__ == '__main__':
	app.run(host="192.168.1.102",port="107")
