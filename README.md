# face-recognition-program-use-transfer-learning
face recognition program use transfer learning 
first we have to install some library (tensorflow,keras,numpy,glob,matplotlib,opencv,pillow)


i use vgg16 for pretrained weights and keras has vgg16 model as object
<img width="900" height="507" src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png" class="attachment-full size-full wp-post-image" alt="vgg16">
<p>VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”.<a id="bib" href="https://arxiv.org/abs/1409.1556" rel="noopener nofollow external" data-wpel-link="external" target="_blank"></a> The model achieves 92.7% top-5 test accuracy in ImageNet,<a id="bib" href="http://image-net.org/" rel="noopener nofollow external" data-wpel-link="external" target="_blank"></a> which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous model submitted to <a href="http://www.image-net.org/challenges/LSVRC/2014/results" data-wpel-link="external" target="_blank" rel="nofollow external noopener">ILSVRC-2014</a>. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.</p>
<img class="aligncenter size-full wp-image-5385" src="https://neurohive.io/wp-content/uploads/2018/11/vgg16.png" alt="vgg16 architecture " width="1200" height="294">
===========================================================================================================================================

# transfer_learning.py

In this i add preprocessing layer to the front of VGG  and than i cut the last layer vgg and add my softmax layer 


<code>vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False) </code>
  
    
  <br>
To don't train existing weights i put false in vgg which creat up


<code>layer.trainable = False</code>
  
  <br>
by this i add the sotfmax layer in my model
  
  
<code>prediction = Dense(len(folders), activation='softmax')(x)
  
model = Model(inputs=vgg.input, outputs=prediction)e</code>

  
  <br>
  
  and by this i save the model in file
  <code>model.save('facefeatures_new_model.h5')</code>
  
  
 #face_detect_img.py
  
  Basically it's used the model and give prediction to us by just passing image
  
  In this i fisrt check the face in image by haarcascade_frontalface_default.xml
  
  and than by just  passing the image array in model it give the prediction of each class 
  
 In prediction it give list of every class how much similer are in ther 
 
 
#face_detect_live_cam.py
 
 it do same process as face_detect_img.py it but it get image form you defult laptop camera or any other camera by just passing the  image
 
 
 
#facecut.py
In this it cut the face and save it in same file  

It do the same with whole data set
<img src="exampleimg/fullimage.jpg" >


<img src="exampleimg/cutimage.jpg" >

this type of image will use in dataset 


and i get this much accuracy and loss for five class

'val_loss': [0.35284554958343506, 0.39085084199905396, 0.07268141210079193, 0.28053680062294006, 0.1134299784898758], 

'val_accuracy': [0.8461538553237915, 0.807692289352417, 0.9615384340286255, 0.8461538553237915, 0.9615384340286255], 

'loss': [1.482390836433128, 0.5041286514865028, 0.23201437404862157, 0.14587565042354442, 0.10501196042255119], 

'accuracy': [0.5138889, 0.8333333, 0.9351852, 0.9444444, 0.9722222]

<img src="exampleimg/accimage.jpg" >

<img src="exampleimg/lossimage.jpg" >


  
