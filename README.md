# AICITY2025
---

## Clone
```python
git clone https://github.com/k19tvan/AICITY2025
cd AICITY2025
```
## Build Docker Image
```python
docker build -t aicity2025 .
```

## Create Docker Container
```python
docker run -dit --name aicity2025  -v ./:/workspace  --gpus all --ipc=host aicity2025 tail -f /dev/null
```
## Data Preparation
```python
chmod +x download_data.sh download_model.sh
./download_data.sh
./download_model.sh
```
## Yolo - Training
```python
chmod +x train_yolo.sh
./train_yolo.sh
```
## Dfine - Training
```python
chmod +x train_dfine.sh
./train_dfine.sh
```