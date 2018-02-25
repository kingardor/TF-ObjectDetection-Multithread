cd object_detection
mkdir ssd_mobilenet_v1_coco_11_06_2017 && cd ssd_mobilenet_v1_coco_11_06_2017
wget -O frozen_inference_graph.pb https://www.dropbox.com/s/0rrh4jblsbrwzjg/frozen_inference_graph.pb?dl=1
cd ../..
echo "Download of frozen_inference_graph.pb for object detection successful"
