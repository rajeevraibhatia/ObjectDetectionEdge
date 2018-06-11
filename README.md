# Object Detection using the TensorFlow  (Mast Detection)

Project: 
Duration: 15th June – 30th July 2017
Completed by: Rajeev Rai Bhatia
Technology Stack: Python, TensorFlow, OpenCV

Background:
The idea of the project is to create a real-time object detection model to be deployed on an NVidia Tx1 on-board a drone for real-time object detection for Mast Detection.

How to Re-train a pre-trained Model using a new Dataset on the Server:
Follow the tutorial here https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md

How to Build your own Model for TensorFlow:
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/defining_your_own_model.md

Steps to train a Model for Mast Detection at eSmart Systems Nvidia Titan X Server:

1.	Run Docker Container

nvidia-docker  run -it \
  --publish 8888:6006 \
  --volume ${HOME}/models:/models \
  --workdir /models \
rrb-1.2.1-gpu-v2 bash

2.	Goto object_detection folder

cd /models/object_detection

3.	Run the protobuf / slim commands to prepare object detection for training / running scripts

protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

Execute	below command to check if build is ok – then we are ready to run training/eval and
TensorFlow object detection scripts

python object_detection/builders/model_builder_test.py

4.	Create Mast Files if doing for the first time only or a new dataset (Not needed for retraining the existing model)

python create_mast_tf_record.py --data_dir=`pwd`/Mastdevkit --output_dir=`pwd`/Mastdevkit

python create_pascal_tf_record.py --data_dir=Mastdevkit \
    --year=VOC2007 --set=train --output_path=mast_train.record

5.	Run Training Script

python eval.py \
    --logtostderr \
    --pipeline_config_path=ssd_mobilenet_v1_mast.config \
    --checkpoint_dir=masttrain \
    --eval_dir=masteval

6.	Run Eval Script (Check TensorBoard output on the server IP and docker port to monitor the training and eval results during the training. Can also plot a graph for accuracy estimation)

python eval.py \
    --logtostderr \
    --pipeline_config_path=ssd_mobilenet_v1_mast.config \
    --checkpoint_dir=masttrain \
    --eval_dir=masteval

7.	Export Inference Graph

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ssd_mobilenet_v1_mast.config \
    --checkpoint_path masttrain/model.ckpt-163997 \
    --inference_graph_path masttrain/resulting_inference_graph.pb


8.	The resulting graph can then be used for testing using below code:

https://github.com/rajeevraibhatia/ObjectDetectionEdge
https://github.com/tensorflow/models/tree/master/object_detection

Clone both the above in your local machine with below folder structure:



Install TensorFlow, OpenCV, Python on your machine or use the Docker image from the server. You can also use Anaconda (environment.yml) to create the environment on your local for all the libraries. 


9.	Save the label map resulting_inference_graph.pb from the server to the local machine inside object_detection folder (path mentioned in run_object_detection.py

10.	Save the mast_label_map.pbtxt from server to object_detection/data folder

11.	Check Python, TensorFlow and OpenCV installations before running the code


12.	Run the below command after you connect a webcam to source 0 (width, height and source is configurable for OpenCV real time stream):

python run_object_detection.py --width=1024 --height=768 --source-0

	This should open a web camera live feed which you can point to sample images from the dataset. 
