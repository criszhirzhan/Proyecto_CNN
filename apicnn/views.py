from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import json

#from tensorflow import Graph, Session





img_height, img_width=32,32
with open('./models/clases.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)


#Los gráficos son utilizados por tf.functions para representar los cálculos de la función.
# Cada gráfico contiene un conjunto de tf.Operationobjetos, que representan unidades de cálculo; y
#  tf.Tensorobjetos, que representan las unidades de datos que fluyen entre operaciones.
model_graph = tf.Graph()
with model_graph.as_default():
    #tf.compat.v1.Session() proporciona el entorno para ejecutar objetos tf.Operation y evaluar objetos tf.Tensor. 
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/modeloCNN.h5')



def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):
    import numpy as np
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)

            #Se transforma la probabilidad sobre el 100%
            probabilidad=predi[:,np.argmax(model.predict(x))]
            probabilidad=probabilidad*100
            

    

    
    #CATEGORIAS 'avion', 'automovil','pajaro','gato','ciervo','perro','rana','caballo','embarcacion','camion' 
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel[1], 'Probabilidad':probabilidad}
    return render(request,'index.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 