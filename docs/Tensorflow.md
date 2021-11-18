<pre>
sudo apt-get update
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools==49.6.0 
sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
</pre>






2021-11-17 22:16:42.180100: W tensorflow/core/common_runtime/bfc_allocator.cc:274] Allocator (GPU_0_bfc) ran out of memory trying to allocate 16.00MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.

frozen_inference_graph.pb, is a frozen graph that cannot be trained anymore, it defines the graphdef and is actually a serialized graph
Esta dentro del modulo de detección de objetos ssdlite_mobilenet_v2_coco_2018_05_09

mscoco_label_map.pbtxt contiene los objetos que puede detectar en el siguiente formato:
item {
  name: "/m/015qff"
  id: 10
  display_name: "traffic light"
}
