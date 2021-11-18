
# Instalación de TensorFlow
<pre>
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools==49.6.0 
sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
</pre>


Probamos TensorFlow con un [grafo congelado que ya no se puede entrenar](https://github.com/jmvega/tfg-amariscal/blob/main/src/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb), dicho grafo se encuentra dentro del módulo de detección de objetos [ssdlite_mobilenet_v2_coco_2018_05_09](https://github.com/jmvega/tfg-amariscal/tree/main/src/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09)

Y las etiquetas se encuentran [aquí](https://github.com/jmvega/tfg-amariscal/blob/main/src/object_detection/data/mscoco_label_map.pbtxt), podemos ver que se encuentran todos objetos que puede detectar, en el siguiente formato:

<pre>
item {
  name: "/m/015qff"
  id: 10
  display_name: "traffic light"
}
</pre>

Hemos realizado el siguiente [programa](https://github.com/jmvega/tfg-amariscal/blob/main/src/object_detection/objectDetectionTensorflow.py) modificado a partir de [este](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/master/Object_detection_picamera.py), se trataba de un programa para detectar objetos en Raspberry Pi con la Pi Camera o una cámara USB, lo hemos modificado para utilizar únicamente un frame obtenido por la siguiente imagen de una ciudad:

![](https://github.com/jmvega/tfg-amariscal/blob/main/resources/city.jpg)

Vemos que detecta los semáforos, las personas y los coches. Las personas las detecta con probabilidad baja, alrededor de 50%

![](https://github.com/jmvega/tfg-amariscal/blob/main/resources/savedImage.jpg)

Vemos el tiempo que tarda:

![](https://github.com/jmvega/tfg-amariscal/blob/main/resources/time.png)

La mayoría del tiempo se invierte en abrir las librerías de CUDA y TensorFlow

A continuación medimos el tiempo real que tarda únicamente en procesar la imagen, sin contar el tiempo en cargar las librerías:

![](https://github.com/jmvega/tfg-amariscal/blob/main/resources/time.png)


Tras ejecutar varias veces el programa me salta el siguiente mensaje:

<pre>
2021-11-17 22:16:42.180100: W tensorflow/core/common_runtime/bfc_allocator.cc:274] Allocator (GPU_0_bfc) ran out of memory trying to allocate 16.00MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
</pre>

Parece que se queda sin memoria o no la libera correctamente

Htop en el momento de ejecutar el programa:

![](https://github.com/jmvega/tfg-amariscal/blob/main/resources/htop.png)

Vemos que la memoria virtual utilizada por el programa es 13 GB
