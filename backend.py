from flask import Flask,request,render_template,send_file
import tensorflow as tf
import tensorflow_addons as tfa
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

app=Flask(__name__)

def encode(filters,size,apply_instancenorm=True):
  initial_filter=tf.random_normal_initializer(0,0.02)
  model=tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(filters,size,padding="same",strides=2,kernel_initializer=initial_filter,use_bias=False))
  if apply_instancenorm:
    model.add(tfa.layers.InstanceNormalization())
  model.add(tf.keras.layers.LeakyReLU())
  return model

def decode(filters,size,apply_dropout=False):
  initial_filter=tf.random_normal_initializer(0,0.02)
  model=tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2DTranspose(filters,size,padding="same",strides=2,kernel_initializer=initial_filter,use_bias=False))
  model.add(tfa.layers.InstanceNormalization())
  if apply_dropout:
    model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.LeakyReLU())
  return model

def generator_downsample(inputs,downsampling):
  x=inputs
  stores=[]
  for downsample in downsampling:
    x=downsample(x)
    stores.append(x)
  stores=stores[:-1]
  stores=reversed(stores)
  return x,stores  

def generator_upsample(inputs,upsampling,stores):
  x=inputs
  for upsample,store in zip(upsampling,stores):
    x=upsample(x)
    x=tf.keras.layers.Concatenate()([x,store])
  return x  

def Generator():
  inputs=tf.keras.layers.Input(shape=[256,256,3])

  downsampling=[
      encode(64,4,apply_instancenorm=False), #(batch_size,128,128,64)
      encode(128,4),                      #(batch_size,64,64,128)
      encode(256,4),                      #(batch_size,32,32,256)
      encode(512,4),                      #(batch_size,16,16,512)
      encode(512,4),                      #(batch_size,8,8,512)
      encode(512,4),                      #(batch_size,4,4,512)
      encode(512,4),                      #(batch_size,2,2,512)
      encode(512,4)                       #(batch_size,1,1,512)
  ]
  upsampling=[
      decode(512,4,apply_dropout=True),    #(batch_size,2,2,512)
      decode(512,4,apply_dropout=True),    #(batch_size,4,4,512)
      decode(512,4,apply_dropout=True),    #(batch_size,8,8,512)
      decode(512,4),                       #(batch_size,16,16,512)
      decode(256,4),                       #(batch_size,32,32,256)
      decode(128,4),                       #(batch_size,64,64,128)
      decode(64,4)                         #(batch_size,128,128,64)
  ]

  initial_filter=tf.random_normal_initializer(0,0.02)
  last=tf.keras.layers.Conv2DTranspose(filters=3,kernel_size=4,padding="same",strides=2,kernel_initializer=initial_filter,activation="tanh")

  x,stores=generator_downsample(inputs,downsampling)

  x=generator_upsample(x,upsampling,stores)

  x=last(x)

  return tf.keras.Model(inputs = inputs, outputs = x)


generator_f_cezanne=Generator()
generator_f_vangogh=Generator()
generator_f_ukiyoe=Generator()

generator_f_cezanne.load_weights("D:\model_cezanne_epoch_70 .h5")
generator_f_vangogh.load_weights("D:\model_vangogh_epoch_70.h5")
generator_f_ukiyoe.load_weights("D:\model_ukiyoe_epoch_70.h5")

def process_testing_img(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img)
    img = tf.convert_to_tensor(img)
    img=tf.image.resize(img,size=[256,256])
    img = (img / 127.5) - 1
    return img

def generate_images(input_img,generator):
    img=process_testing_img(input_img)
    img=tf.expand_dims(img,axis=0)
    generated_image=generator(img,training=False)
    generated_image = (generated_image[0] + 1) * 127.5  
    generated_image = tf.cast(generated_image, tf.uint8)

    return Image.fromarray(generated_image.numpy())
    
    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/style_transfer",methods=["POST"])
def style_transfer():
    style=request.form["style"]
    image_file=request.files["image"]
    
    img=Image.open(image_file.stream)
    
    if style == 'cezanne':
        generator = generator_f_cezanne
    elif style == 'vangogh':
        generator = generator_f_vangogh
    elif style == 'ukiyoe':
        generator = generator_f_ukiyoe
    
    output=generate_images(img,generator)
    
    buffer = io.BytesIO()
    output.save(buffer, format='PNG')
    buffer.seek(0)
    
    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

    
        
    
    
    
    
    
    
    
    
    
    


 
    
    
