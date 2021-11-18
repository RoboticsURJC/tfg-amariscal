2021-11-17 22:16:42.180100: W tensorflow/core/common_runtime/bfc_allocator.cc:274] Allocator (GPU_0_bfc) ran out of memory trying to allocate 16.00MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.

frozen_inference_graph.pb, is a frozen graph that cannot be trained anymore, it defines the graphdef and is actually a serialized graph
Esta dentro del modulo de detección de objetos ssdlite_mobilenet_v2_coco_2018_05_09

mscoco_label_map.pbtxt contiene los objetos que puede detectar en el siguiente formato:
item {
  name: "/m/015qff"
  id: 10
  display_name: "traffic light"
}
