# object-detection-api for tf1.15

Make it easy to train and deploy Object Detection(SSD) and Image Segmentation(Mask R-CNN) Model Using [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

[TF1.15可用的预练习模型](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)

# Tensorflow 对象识别迁移学习过程如下

## 环境要求

- [conda](https://docs.conda.io/en/latest/miniconda.html)
- [protoc>3.0](https://github.com/protocolbuffers/protobuf)
- [tensorflow=1.15](https://github.com/tensorflow/tensorflow)

## 安装

### Conda

```shell
$ conda create -n  od python=3.6 tensorflow=1.15 protobuf=3.17 && conda activate od && make install
......
----------------------------------------------------------------------
Ran 24 tests in 21.869s

OK (skipped=1)
```

### Bazel

```shell
cd "/usr/local/lib/bazel/bin" && curl -fLO https://mirrors.huaweicloud.com/bazel/5.3.0/bazel-5.3.0-linux-x86_64 && chmod +x bazel-5.3.0-linux-x86_64  &&
ln -sf /usr/local/lib/bazel/bin/bazel-5.3.0-linux-x86_64  /usr/local/bin/bazel &&
bazel --version
```

### 安装对象识别模型及tf1.15的python环境

```shell
$ make install
......
----------------------------------------------------------------------
Ran 24 tests in 21.869s

OK (skipped=1)
```

## 创建工作空间

在`object-detection-api`根目录下执行:

```shell
make workspace-box SAVE_DIR=workspace NAME=test
```

## 上传标注文件,并转换成Tfrecord

进入`workspace/test`目录,将您标注好的`jpg文件`和`xml文件`上传到images目录,

在`test`根目录下执行:

```shell
make gen-tfrecord
```

将在 `annotations` 文件夹下生成 `label_map.pbtxt` 和 [TFRecord](https://links.jianshu.com/go?to=https%3A%2F%2Fwww.tensorflow.org%2Fapi_docs%2Fpython%2Ftf%2Fdata%2FTFRecordDataset) 格式的数据集

```shell
annotations
├── label_map.pbtxt
├── train.record
└── val.record
```

## 下载预训练模型

在 `test`根目录下执行:

```shell
make dl-model
```

## 配置训练Pipeline

在目录`models`下创建文件夹`ssd_mobilenet_v2_coco_2018_03_29`,在此目录我们将文件`models\research\object_detection\samples\configs\ssd_mobilenet_v2_coco.config`移动到该目录下

```
/home/object-detection-api/workspace/test/models/ssd_mobilenet_v2_coco_2018_03_29/
├── pipeline.config
├── ssd_mobilenet_v2_coco.config
```

`─ ssd_mobilenet_v2_coco.conf`有几处需要修改:

```
model {
  ssd {
    num_classes: 3 # 修改为需要识别的目标个数，示例项目为 3 种
    ......
}
train_config {
  batch_size: 8 # 这里需要根据自己的配置，调整大小，这里设置为 8,配置好往高了调
  ......
  fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0" # 修改为预制模型的路径
  fine_tune_checkpoint_type:  "detection"
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: 200000 # 训练的总步数,也可以在训练make train-t1 命令那修改
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}
train_input_reader {
  label_map_path: "annotations/label_map.pbtxt" # 修改为标注的路径
  tf_record_input_reader {
    input_path: "annotations/train.record" # 修改为训练集的路径
  }
}
eval_config: {
  num_examples: 8000
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 10
}
eval_input_reader {
  label_map_path: "annotations/label_map.pbtxt" # 修改为标注的路径
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "annotations/val.record" # 修改为验证集的路径
  }
}
```

## 训练模型

```shell
make train
```

最后输出目录如下:

```
├── train
│   ├── checkpoint
│   ├── eval_0
│   │   └── events.out.tfevents.1663666310.2a48bf3f98bf
│   ├── events.out.tfevents.1663665669.2a48bf3f98bf
│   ├── export
│   │   └── Servo
│   │       └── 1663672202
│   │           ├── saved_model.pb
│   │           └── variables
│   │               ├── variables.data-00000-of-00001
│   │               └── variables.index
│   ├── graph.pbtxt
│   ├── model.ckpt-10000.data-00000-of-00001
│   ├── model.ckpt-10000.index
│   ├── model.ckpt-10000.meta
│   ├── model.ckpt-6527.data-00000-of-00001
│   ├── model.ckpt-6527.index
│   ├── model.ckpt-6527.meta
│   ├── model.ckpt-7438.data-00000-of-00001
│   ├── model.ckpt-7438.index
│   ├── model.ckpt-7438.meta
│   ├── model.ckpt-8351.data-00000-of-00001
│   ├── model.ckpt-8351.index
│   ├── model.ckpt-8351.meta
│   ├── model.ckpt-9262.data-00000-of-00001
│   ├── model.ckpt-9262.index
│   └── model.ckpt-9262.meta
```

## 模型导出

```shell
make export-t1
```

输出目录:

```
home/object-detection-api/workspace/test/exported-models/ssd_mobilenet_v2_coco_2018_03_29
├── checkpoint
├── frozen_inference_graph.pb
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
├── pipeline.config
├── saved_model
│   └── variables
├── saved_model.pb

```

## 将训练后的TF模型转为mediapipe上移动端可用的模型

如何确定  make convert-lite-t1 命令内部的参数?

首先在执行make export-lite-t1后会生成两个文件

```
```
/home/object-detection-api/workspace/test/exported-models/ssd_mobilenet_v2_coco_2018_03_29
├── ***
├── tflite_graph.pb
└── tflite_graph.pbtxt
```
```

我们可以使用`summarize_graph`工具获取图形的输入和输出

```shell
git clone https://github.com/tensorflow/tensorflow.git
```

在tensorflow主目录下执行:

```shell
bazel run tensorflow/tools/graph_transforms:summarize_graph -- \
--in_graph=${PATH_TO_MODEL}/tflite_graph.pb
```

输出内容为:

```
Found 1 possible inputs: (name=normalized_input_image_tensor, type=float(1), shape=[1,300,300,3]) 
No variables spotted.
Found 2 possible outputs: (name=raw_outputs/box_encodings, op=Identity) (name=raw_outputs/class_predictions, op=Identity) 
Found 4656295 (4.66M) const parameters, 0 (0) variable parameters, and 0 control_edges
Op types used: 411 Identity, 340 Const, 60 FusedBatchNormV3, 55 Conv2D, 43 Relu6, 17 DepthwiseConv2dNative, 12 BiasAdd, 12 Reshape, 10 AddV2, 2 ConcatV2, 1 Mul, 1 Placeholder, 1 Sigmoid, 1 Squeeze
```

这样你可以发现例子模型的输入是300X300,输出是

- raw_outputs/box_encodings

- raw_outputs/class_predictions

然后有以上数据,我们就可以转换tflite模型了,`make convert-lite-t1`同以下命令:

```
tflite_convert --  \
  --graph_def_file=${PATH_TO_MODEL}/tflite_graph.pb \
  --output_file=${PATH_TO_MODEL}/model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shapes=1,320,320,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=raw_outputs/box_encodings,raw_outputs/class_predictions
```

恭喜,最终您将获得您第一个TFLite 模型 `mode.tflite`,可以应用在mediapipe了.

## Usage

#### Object Detection

[The easiest way to Train a Custom Object Detection Model Using TensorFlow Object Detection API](https://makeoptim.com/en/deep-learning/yiai-object-detection)

#### Image Segmentation

[The easiest way to Train a Custom Image Segmentation Model Using TensorFlow Object Detection API Mask R-CNN](https://makeoptim.com/en/deep-learning/yiai-image-segmentation)

### Deploy

https://www.jianshu.com/p/0d2ece983bfb

[Deploy image segmentation (Mask R-CNN) model service with TensorFlow Serving & Flask](https://makeoptim.com/en/deep-learning/yiai-serving-flask-mask-rcnn)
