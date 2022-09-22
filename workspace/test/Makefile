gen-tfrecord:
	python gen_label_map.py
	
	python generate_tfrecord.py \
	-x images/train \
	-l annotations/label_map.pbtxt \
	-o annotations/train.record

	python generate_tfrecord.py \
	-x images/val \
	-l annotations/label_map.pbtxt \
	-o annotations/val.record

dl-model-t1:
	mkdir -p pre-trained-models; \
	model=ssd_mobilenet_v2_coco_2018_03_29; \
	curl -O http://download.tensorflow.org/models/object_detection/$$model.tar.gz; \
	tar zxvf $$model.tar.gz; \
	mv -f $$model ./pre-trained-models/; \
	rm -rf $$model $$model.tar.gz;
train-t1:
	python model_main.py \
	--model_dir=models/ssd_mobilenet_v2_coco_2018_03_29/train \
	--pipeline_config_path=models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco.config \
	--num_train_steps=10000
export-t1:
	python export_inference_graph.py \
	--input_type=image_tensor \
	--pipeline_config_path=models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco.config \
	--trained_checkpoint_prefix=models/ssd_mobilenet_v2_coco_2018_03_29/train/model.ckpt-8351 \
	--output_directory exported-models/ssd_mobilenet_v2_coco_2018_03_29
export-lite-t1:
	python export_tflite_ssd_graph.py \
	--pipeline_config_path=exported-models/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config \
	--trained_checkpoint_prefix=exported-models/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt \
	--output_directory=exported-models/ssd_mobilenet_v2_coco_2018_03_29 \
	--add_postprocessing_op=False
convert-lite-t1:
	tflite_convert  \
	--graph_def_file=exported-models/ssd_mobilenet_v2_coco_2018_03_29/tflite_graph.pb \
	--output_file=exported-models/ssd_mobilenet_v2_coco_2018_03_29/model.tflite \
	--input_format=TENSORFLOW_GRAPHDEF \
	--output_format=TFLITE \
	--inference_type=FLOAT \
	--input_shapes=1,300,300,3 \
	--input_arrays=normalized_input_image_tensor \
	--output_arrays=raw_outputs/box_encodings,raw_outputs/class_predictions
dl-model:
	mkdir -p pre-trained-models; \
	model=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8; \
	curl -O http://download.tensorflow.org/models/object_detection/tf2/20200711/$$model.tar.gz; \
	tar zxvf $$model.tar.gz; \
	mv -f $$model ./pre-trained-models/; \
	rm -rf $$model $$model.tar.gz;
dl-model-640:
	model=ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8; \
	curl -O http://download.tensorflow.org/models/object_detection/tf2/20200711/$$model.tar.gz; \
	tar zxvf $$model.tar.gz; \
	mv -f $$model ./pre-trained-models/; \
	rm -rf $$model $$model.tar.gz;
train:
	python model_main_tf2.py \
	--model_dir=models/ssd_mobilenet_v2_fpnlite_320x320 \
	--pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config
train-640:
	python model_main_tf2.py \
	--model_dir=models/ssd_mobilenet_v2_fpnlite_640x640 \
	--pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config
eval:
	python model_main_tf2.py \
    --pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config \
    --model_dir=models/ssd_mobilenet_v2_fpnlite_320x320 \
    --checkpoint_dir=models/ssd_mobilenet_v2_fpnlite_320x320 \
    --alsologtostderr
eval-640:
	python model_main_tf2.py \
    --pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config \
    --model_dir=models/ssd_mobilenet_v2_fpnlite_640x640 \
    --checkpoint_dir=models/ssd_mobilenet_v2_fpnlite_640x640 \
    --alsologtostderr
.PHONY: export
export:
	python exporter_main_v2.py \
	--input_type image_tensor \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_320x320/ \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_320x320
export-640:
	python exporter_main_v2.py \
	--input_type image_tensor \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_640x640/ \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_640x640
export-lite:
	python export_tflite_graph_tf2.py \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_320x320 \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite
export-lite-640:
	python export_tflite_graph_tf2.py \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_640x640 \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite
convert-lite:
	tflite_convert \
	--saved_model_dir=exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite/saved_model \
	--output_file=exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite/detect.tflite
convert-lite-640:
	tflite_convert \
	--saved_model_dir=exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite/saved_model \
	--output_file=exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite/detect.tflite
convert-quant-lite:
	python convert_quant_lite.py \
	--saved_model_dir exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite/saved_model \
	--output_file exported-models/ssd_mobilenet_v2_fpnlite_320x320_quant_lite/detect.tflite
convert-quant-lite-640:
	python convert_quant_lite.py \
	--saved_model_dir exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite/saved_model \
	--output_file exported-models/ssd_mobilenet_v2_fpnlite_640x640_quant_lite/detect.tflite
