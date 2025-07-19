#!/usr/bin/env bash

# Setup Kaggle API credentials
echo "Setting up Kaggle credentials..."
mkdir -p ~/.kaggle
cp /workspace/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Yolo 
echo "Starting yolo dataset download..."
mkdir -p /workspace/datasets/yolo/
cd /workspace/datasets/yolo/
kaggle datasets download whonoac/carlafisheyecubidangcapso1vietname --unzip
echo "Yolo Dataset downloaded to /workspace/yolo/datasets"

# Dfine
echo "Starting dfine dataset download..."
mkdir -p /workspace/datasets/dfine/
cd /workspace/datasets/dfine/

kaggle datasets download -d newnguyen/dfine-batch-1 --unzip
echo "Dfine-Batch-1 downloaded to /workspace/datasets/dfine/batch_1/"

kaggle datasets download -d newnguyen/dfine-batch-2 --unzip 
echo "Dfine-Batch-2 downloaded to /workspace/datasets/dfine/batch_2/"

kaggle datasets download -d newnguyen/dfine-val --unzip 
echo "Dfine-Val downloaded to /workspace/datasets/dfine/val/"

# Stage 2
echo "Starting download stage 2 datasets..."
cd /workspace/datasets/
kaggle datasets download -d newnguyen/yolo-stage-2 --unzip

# To Dfine
cp /workspace/datasets/yolo-stage-2/train/images /workspace/datasets/dfine-stage-2/train/

# Merge Dfine
python src/merge_dfine.py --batch1 /workspace/datasets/dfine/batch_1 --batch2 /workspace/datasets/dfine/batch_2 --output /workspace/datasets/dfine/

# Coco2yolo
python src/yolo2coco.py --images_dir datasets/dfine/train/images --labels_dir datasets/dfine/train/labels --output_json datasets/dfine/train/train.json 
python src/yolo2coco.py --images_dir datasets/dfine/val/images --labels_dir datasets/dfine/val/labels --output_json datasets/dfine/val/val.json 