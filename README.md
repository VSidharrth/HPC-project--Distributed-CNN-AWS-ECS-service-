# Cloud-Based Distributed CNN for Multi-Class Blood Cancer Classification

A High Performance Computing (HPC) and cloud deployment project that uses **MPI-based data parallelism** to train a CNN for multi-class blood cancer classification, followed by **Docker containerization and AWS ECS Fargate deployment** for web-based inference.

---

## Overview

This project implements an end-to-end pipeline for classifying microscopic **Peripheral Blood Smear (PBS)** images into four blood-cell categories associated with Acute Lymphoblastic Leukemia (ALL).

The project combines two major components:

1. **High Performance Computing:** Distributed CNN training using MPI and data parallelism across multiple processes.
2. **Cloud Deployment:** Containerization with Docker, image storage in Amazon ECR, and deployment through Amazon ECS Fargate with an Nginx-based web interface.

The workflow is:

```text
PBS Blood Cell Dataset
        │
        ▼
Data Validation & Preprocessing
        │
        ▼
70% Train / 15% Validation / 15% Test
        │
        ▼
MPI Data Parallelism
        │
        ├── Process 0 ──┐
        └── Process 1 ──┤
                       ▼
             Synchronized CNN Training
                       │
                       ▼
                 Trained Keras Model
                       │
                       ▼
              TensorFlow.js Conversion
                       │
                       ▼
                  Docker + Nginx
                       │
                       ▼
                     AWS ECR
                       │
                       ▼
                 AWS ECS Fargate
                       │
                       ▼
              Browser-based Prediction
```

---

## Problem Statement

Training convolutional neural networks involves repeated forward and backward passes over large image datasets. As model training becomes computationally expensive, parallel processing can distribute the workload across multiple processes.

This project investigates **data parallelism using MPI** for CNN training and then demonstrates how the resulting model can be containerized and deployed as a scalable cloud inference service.

---

## Dataset

The project uses microscopic Peripheral Blood Smear images from **89 patients suspected of Acute Lymphoblastic Leukemia (ALL)**.

The presentation reports a total of **3,242 images** across four classes:

| Class | Images |
|---|---:|
| Benign | 512 |
| Malignant Pre-B | 955 |
| Malignant Pro-B | 796 |
| Malignant Early Pre-B | 979 |
| **Total** | **3,242** |

The classification task is therefore a **4-class image classification problem**.

---

## HPC Approach

### Data Parallelism

The project uses **data parallelism**, where the same CNN model is replicated across MPI processes and each process works on a different portion of the training data.

For example:

```text
                    Training Dataset
                          │
                          ▼
                 ┌─────────────────┐
                 │   Master Rank 0 │
                 └────────┬────────┘
                          │
                     MPI scatter
                    ┌─────┴─────┐
                    ▼           ▼
               Process 0    Process 1
                 Batch         Batch
                    │           │
                    ▼           ▼
                CNN Forward + Backward
                    │           │
                    └─────┬─────┘
                          │
                    MPI Allreduce
                          │
                          ▼
                 Averaged Parameters
                          │
                          ▼
                Synchronized Models
```

### Why Data Parallelism?

The project specifically chose data parallelism rather than model parallelism because:

- The CNN fits within available memory.
- Model parallelism would introduce unnecessary layer-splitting and communication overhead.
- Dataset partitioning is simpler and suitable for the relatively small CNN.
- Synchronizing model parameters across processes provides a practical distributed-training demonstration.

---

## MPI Implementation

The distributed training implementation is contained in:

```text
train_mpi.py
```

The code uses:

```python
from mpi4py import MPI
```

and initializes:

```python
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```

### 1. Dataset preparation

Rank 0:

- Scans the dataset.
- Validates image files.
- Removes invalid/corrupted images.
- Shuffles the dataset.
- Creates train, validation and test partitions.

The split is:

```text
Training     → 70%
Validation   → 15%
Testing      → 15%
```

### 2. Training data distribution

The training dataset is divided between MPI processes using:

```python
comm.scatter()
```

Each process therefore receives a local portion of the training data.

### 3. Model synchronization

The initial model weights are broadcast using:

```python
comm.Bcast()
```

so that every process starts from synchronized parameters.

### 4. Local computation

Each process independently performs:

- Forward propagation
- Loss computation
- Backpropagation
- Local parameter updates

The CNN uses TensorFlow's `GradientTape` for gradient computation.

### 5. Parameter aggregation

After local computation, model weights are synchronized through MPI `Allreduce`.

The summed parameters are averaged across processes:

```text
Global Parameters =
Sum of Parameters Across Processes / Number of Processes
```

A barrier synchronization ensures that processes remain synchronized before proceeding to the next batch.

---

## CNN Architecture

The project uses a custom CNN implemented with TensorFlow/Keras.

```text
Input: 224 × 224 × 3
        │
        ▼
Conv2D (32 filters, 3×3) + ReLU
        │
     MaxPooling
        │
        ▼
Conv2D (64 filters, 3×3) + ReLU
        │
     MaxPooling
        │
        ▼
Conv2D (128 filters, 3×3) + ReLU
        │
     MaxPooling
        │
        ▼
Flatten
        │
        ▼
Dense (256) + ReLU
        │
     Dropout (0.5)
        │
        ▼
Softmax Output
        │
        ▼
4 Classes
```

### Training configuration

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Training split | 70% |
| Validation split | 15% |
| Test split | 15% |

---

## Image Augmentation

Training images use the following augmentation operations:

- Rotation up to 20°
- Width shifting
- Height shifting
- Shearing
- Zooming
- Horizontal flipping
- Brightness variation
- Nearest-neighbor fill

Validation and test images are only rescaled and are not augmented.

---

## Results

The distributed MPI experiment achieved:

| Metric | Result |
|---|---:|
| Validation Accuracy | **90.95%** |
| Test Accuracy | **90.55%** |
| Validation Loss | 0.2752 |
| Test Loss | 0.2688 |
| Training Wall-clock Time | 3468.29 s |
| Training Throughput | 0.654 images/s |
| Test Throughput | 35.42 images/s |

### Test classification performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Benign | 0.8500 | 0.5397 | 0.6602 |
| Malignant Pre-B | 0.9859 | 0.9333 | 0.9589 |
| Malignant Pro-B | 0.9835 | 0.9917 | 0.9876 |
| Malignant Early Pre-B | 0.8043 | 0.9610 | 0.8757 |
| **Overall Accuracy** | | | **0.9055** |

The strongest classification performance is observed for the malignant Pro-B class, while the Benign class has lower recall.

---

## Parallelization Analysis

The project also records epoch-level training time, training loss and accuracy and generates a speedup plot.

The experiment was conducted with **two MPI processes/CPU cores**.

The results showed that parallelization did **not produce meaningful speedup at this scale**. Communication and synchronization overhead were comparable to or greater than the computational benefit.

This is an important HPC observation:

> **Parallelism does not automatically guarantee speedup when the workload is relatively small and communication overhead dominates.**
---

# Cloud Deployment

After training, the model is converted into a browser-compatible format and deployed as a web application.

The deployment pipeline is:

```text
Keras Model (.h5)
      │
      ▼
TensorFlow.js Conversion
      │
      ▼
webs_model/
      │
      ▼
HTML + JavaScript Interface
      │
      ▼
Nginx Docker Container
      │
      ▼
Docker Image
      │
      ▼
Amazon ECR
      │
      ▼
Amazon ECS Fargate
      │
      ▼
Public/Cloud Web Service
```

---

## TensorFlow.js Conversion

The `Conversionscript.ipynb` notebook converts the trained Keras model into TensorFlow.js format.

The conversion uses:

```bash
tensorflowjs_converter \
    --input_format keras \
    blood_cancer_model.h5 \
    webs_model
```

The resulting model can be loaded by the browser-based prediction interface.

---

## Web Interface

The repository contains:

```text
index.html
```

The web interface connects the converted TensorFlow.js model to an image-prediction workflow.

The application predicts one of the four blood-cell classes:

```text
Benign
Malignant Pre-B
Malignant Pro-B
Malignant Early Pre-B
```

---

## Docker Deployment

The project uses an Nginx Alpine image.

The Dockerfile:

```dockerfile
FROM nginx:alpine

COPY ./index.html /usr/share/nginx/html/
COPY ./webs_model /usr/share/nginx/html/webs_model

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

Build the image:

```bash
docker build -t second .
```

Run locally:

```bash
docker run -d -p 8085:80 second
```

The application can then be accessed through:

```text
http://localhost:8085/index.html
```

---

# AWS Deployment

## Amazon ECR

The Docker image is pushed to **Amazon Elastic Container Registry (ECR)**.

The deployment workflow is:

```text
Local Docker Image
        │
        ▼
docker save
        │
        ▼
AWS Cloud Shell
        │
        ▼
docker load
        │
        ▼
Amazon ECR Repository
        │
        ▼
docker push
```

Example commands used in the project:

```bash
docker save -o second.tar second:latest

docker load -i second.tar
```

Then authenticate with ECR and push the image to the repository.

---

## Amazon ECS Fargate

The container is deployed using **Amazon Elastic Container Service (ECS)** with **AWS Fargate**.

The deployment consists of:

1. Creating an ECS cluster.
2. Selecting AWS Fargate as the infrastructure.
3. Creating a task definition.
4. Connecting the task definition to the ECR image.
5. Configuring container ports.
6. Creating an ECS service.
7. Configuring networking and security groups.
8. Running the container as an ECS task.
9. Accessing the application through the task's public IP and configured port.

### Deployment architecture

```text
                    AWS Cloud
                       │
             ┌─────────┴─────────┐
             │                   │
          Amazon ECR         Amazon ECS
             │                   │
        Docker Image       ECS Fargate
                                 │
                              Service
                                 │
                               Task
                                 │
                           Nginx Container
                                 │
                                 ▼
                         Web Prediction App
```

---

## Monitoring

AWS CloudWatch screenshots demonstrating cloud-service monitoring are also included.

Relevant repository assets include:

```text
cloudwatch1.jpg
cloudwatchmetrics.jpg
cluster.jpg
deployment.png
```

---

## Repository Structure

```text
HPC-project--Distributed-CNN-AWS-ECS-service-/
│
├── train_mpi.py
├── classical.py
├── Conversionscript.ipynb
│
├── Dockerfile
├── index.html
├── requirements.txt
│
├── class_names.json
├── metrics_summary.json
├── evaluation_results.txt
├── classification_report_test.csv
├── epoch_times.csv
│
├── accuracy_per_epoch.png
├── loss_per_epoch.png
├── speedup.png
├── classification.png
├── confusion_matrix_val.png
├── confusion_matrix_test.png
│
├── cloudwatch1.jpg
├── cloudwatchmetrics.jpg
├── cluster.jpg
├── deployment.png
│
└── HPCCC_endsem.pptx
```

### File guide

| File | Description |
|---|---|
| `train_mpi.py` | MPI-based distributed CNN training |
| `classical.py` | Non-parallel CNN training baseline |
| `Conversionscript.ipynb` | Keras → TensorFlow.js model conversion |
| `Dockerfile` | Nginx-based Docker deployment |
| `index.html` | Browser-based prediction interface |
| `requirements.txt` | Python dependencies |
| `classification_report_test.csv` | Test-set classification metrics |
| `metrics_summary.json` | Training/evaluation summary |
| `epoch_times.csv` | Epoch-level timing and accuracy |
| `speedup.png` | MPI speedup visualization |
| `accuracy_per_epoch.png` | Training accuracy curve |
| `loss_per_epoch.png` | Training loss curve |
| `confusion_matrix_*.png` | Validation/test confusion matrices |
| `cloudwatch*.jpg` | AWS monitoring screenshots |
| `cluster.jpg` | ECS cluster screenshot |
| `deployment.png` | Deployment-related visualization |
| `HPCCC_endsem.pptx` | Project presentation |

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/VSidharrth/HPC-project--Distributed-CNN-AWS-ECS-service-.git
cd HPC-project--Distributed-CNN-AWS-ECS-service-
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the dataset path

The training scripts currently contain local Windows dataset paths. Update:

```python
DATASET_PATH
```

and the relevant model-save paths to match your environment.

---

## Running the Classical Baseline

The non-parallel implementation is provided in:

```text
classical.py
```

Run:

```bash
python classical.py
```

This trains the same general CNN architecture without MPI-based distributed processing and produces evaluation metrics and plots.

---

## Running MPI Training

MPI training is implemented in:

```text
train_mpi.py
```

An MPI installation compatible with `mpi4py` is required.

For example, with two processes:

```bash
mpiexec -n 2 python train_mpi.py
```

The exact MPI command may vary depending on the operating system and MPI implementation.

---

## Key Technologies

### Deep Learning
- TensorFlow
- Keras
- CNN
- ImageDataGenerator

### High Performance Computing
- MPI
- `mpi4py`
- Data parallelism
- Synchronous parameter aggregation
- Multi-process training

### Cloud & Deployment
- Docker
- Nginx
- Amazon ECR
- Amazon ECS
- AWS Fargate
- CloudWatch

### Data & Evaluation
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Matplotlib
- Seaborn

---

## Future Work

Possible extensions include:

- Evaluating MPI training with 4–8 or more processes.
- Measuring strong and weak scaling.
- Comparing MPI with other distributed-training approaches.
- Using larger CNN architectures.
- Evaluating GPU-based distributed training.
- Investigating communication-efficient synchronization.
- Deploying multiple ECS tasks for scalable inference.
- Adding load balancing and production-grade monitoring.

---

## Outcomes

This project demonstrates a complete pipeline connecting **High Performance Computing, deep learning, containerization, and cloud deployment**:

```text
Parallel Training
       ↓
MPI Data Parallelism
       ↓
CNN Blood Cancer Classifier
       ↓
Model Evaluation
       ↓
TensorFlow.js Conversion
       ↓
Docker Containerization
       ↓
Amazon ECR
       ↓
Amazon ECS Fargate
       ↓
Cloud-Based Image Prediction
```

The most important HPC finding is that **the two-process experiment did not provide significant speedup because communication overhead dominated at small scale**, while the trained model was successfully converted, containerized, and deployed through AWS ECS Fargate.

---

## Authors

**V. Sidharrth**  
**B. G. Rajath Siddarth**  
**Paruchuri Sai**

B.Tech Artificial Intelligence and Data Science  
Amrita Vishwa Vidyapeetham, Bengaluru
---
