import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
import json
import warnings
from PIL import Image

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from mpi4py import MPI
from tqdm import tqdm

# ---- MPI init ----
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---- Config ----
DATASET_PATH = r"C:\Users\HP\Desktop\HPC_Project\Blood cell Cancer [ALL]"
SAVE_DIR = r"C:\Users\HP\Desktop\HPC_Project\web_model"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
TARGET_SIZE = (224, 224)
EPOCHS = 20

# ---- Dataset scanning (rank 0 only) ----
if rank == 0:
    print("Scanning dataset...")
    classes_dirs = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    data = []
    for cls in classes_dirs:
        cls_path = os.path.join(DATASET_PATH, cls)
        for fname in os.listdir(cls_path):
            fp = os.path.join(cls_path, fname)
            try:
                with Image.open(fp) as im:
                    im.verify()
                data.append([fp, cls])
            except Exception as e:
                print(f"Skipping invalid image: {fp} ({e})")
    df = pd.DataFrame(data, columns=["filepath", "class"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Total valid images:", len(df))
    print("Class distribution:\n", df["class"].value_counts())
else:
    df = None
    classes_dirs = None

df_len = np.array([len(df)]) if rank == 0 else np.array([0])
comm.Bcast(df_len, root=0)

# ---- Train / Val / Test split ----
if rank == 0:
    train_frac = 0.7
    val_frac = 0.15
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    df_train = df.iloc[:train_end].reset_index(drop=True)
    df_val = df.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)
    print(f"Split -> train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
else:
    df_train = None
    df_val = None
    df_test = None

# ---- Scatter training data ----
if rank == 0:
    splits = np.array_split(df_train, size)
else:
    splits = None

df_train_chunk = comm.scatter(splits, root=0)

if rank == 0:
    val_df_local = df_val.copy()
    test_df_local = df_test.copy()
else:
    val_df_local = None
    test_df_local = None

# ---- Generators ----
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_dataframe(
    df_train_chunk, x_col="filepath", y_col="class",
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True
)
if rank == 0:
    val_generator = val_datagen.flow_from_dataframe(
        val_df_local, x_col="filepath", y_col="class",
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False
    )
    test_generator = test_datagen.flow_from_dataframe(
        test_df_local, x_col="filepath", y_col="class",
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False
    )

# ---- Model ----
def build_model(n_classes):
    return Sequential([
        Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
        Conv2D(32, 3, activation='relu'), MaxPooling2D(2),
        Conv2D(64, 3, activation='relu'), MaxPooling2D(2),
        Conv2D(128, 3, activation='relu'), MaxPooling2D(2),
        Flatten(),
        Dense(256, activation='relu'), Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])

# ---- Class sync ----
if rank == 0:
    class_list = sorted(df["class"].unique())
else:
    class_list = None
class_list = comm.bcast(class_list, root=0)
n_classes = len(class_list)

model = build_model(n_classes)
optimizer = tf.keras.optimizers.Adam()

weights = model.get_weights()
for i in range(len(weights)):
    comm.Bcast(weights[i], root=0)
model.set_weights(weights)

local_steps = len(train_generator)
steps_per_epoch = int(comm.allreduce(np.array(local_steps, dtype=np.int32), op=MPI.MIN))
if steps_per_epoch == 0:
    steps_per_epoch = local_steps
if rank == 0:
    print("Computed steps_per_epoch:", steps_per_epoch)

epoch_log = []
train_losses = []
train_accs = []

train_start_time_local = time.time()

#  TRAINING LOOP 
for epoch in range(EPOCHS):
    comm.Barrier()
    epoch_start_local = time.time()

    local_losses = []
    local_accs = []

    if rank == 0:
        print(f"\nStarting epoch {epoch+1}/{EPOCHS}")
        progress = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}", unit="batch")

    for step in range(steps_per_epoch):
        X_np, y_np = next(train_generator)
        X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)
        y_tf = tf.convert_to_tensor(y_np, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = model(X_tf, training=True)
            loss_batch = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_tf, logits))

        grads = tape.gradient(loss_batch, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Allreduce weights
        local_weights = model.get_weights()
        avg_weights = []
        for w in local_weights:
            buf = np.zeros_like(w)
            comm.Allreduce(w, buf, op=MPI.SUM)
            avg_weights.append(buf / size)
        model.set_weights(avg_weights)
        comm.Barrier()

        preds = tf.argmax(logits, axis=1).numpy()
        trues = np.argmax(y_np, axis=1)
        local_losses.append(float(loss_batch.numpy()))
        local_accs.append(np.mean(preds == trues))

        if rank == 0:
            progress.update(1)

    if rank == 0:
        progress.close()

    # ---- Aggregate metrics ----
    local_loss_sum = np.sum(local_losses)
    local_acc_sum = np.sum(local_accs)
    local_batches = len(local_losses)

    global_loss_sum = comm.allreduce(local_loss_sum, op=MPI.SUM)
    global_acc_sum = comm.allreduce(local_acc_sum, op=MPI.SUM)
    total_batches = comm.allreduce(local_batches, op=MPI.SUM)

    global_loss = global_loss_sum / total_batches
    global_acc = global_acc_sum / total_batches

    epoch_time_local = time.time() - epoch_start_local
    epoch_time_sum = comm.allreduce(np.array(epoch_time_local, dtype=np.float32), op=MPI.SUM)
    avg_epoch_time = float(epoch_time_sum) / size

    if rank == 0:
        epoch_log.append([epoch+1, avg_epoch_time, global_loss, global_acc])
        train_losses.append(global_loss)
        train_accs.append(global_acc)
        print(f"Epoch {epoch+1} avg time: {avg_epoch_time:.2f}s | Loss={global_loss:.4f} | Acc={global_acc:.4f}")

# gather per-process times
train_end_time_local = time.time()
local_total_train_time = train_end_time_local - train_start_time_local
all_train_times = comm.gather(local_total_train_time, root=0)

if rank == 0:
    total_training_wallclock = float(np.max(all_train_times))
    print(f"\nTotal training wall-clock time (max across ranks): {total_training_wallclock:.2f}s")

    df_epochs = pd.DataFrame(epoch_log, columns=["epoch", "epoch_time_s", "loss", "accuracy"])
    df_epochs.to_csv("epoch_times.csv", index=False)
    print("Saved epoch_times.csv")

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    print("\nEvaluating on validation set...")
    val_start = time.time()
    val_loss, val_acc = model.evaluate(val_generator, verbose=1)
    val_time = time.time() - val_start
    print(f"Validation -> loss: {val_loss:.4f}, acc: {val_acc:.4f}, time: {val_time:.2f}s")

    print("\nEvaluating on test set...")
    test_start = time.time()
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    test_time = time.time() - test_start
    print(f"Test -> loss: {test_loss:.4f}, acc: {test_acc:.4f}, time: {test_time:.2f}s")

    # predictions
    preds_val_probs = model.predict(val_generator, verbose=0)
    preds_val = np.argmax(preds_val_probs, axis=1)
    y_val_true = val_generator.classes

    preds_test_probs = model.predict(test_generator, verbose=0)
    preds_test = np.argmax(preds_test_probs, axis=1)
    y_test_true = test_generator.classes

    class_labels = list(val_generator.class_indices.keys())

    cm_val = confusion_matrix(y_val_true, preds_val)
    cm_test = confusion_matrix(y_test_true, preds_test)

    creport = classification_report(y_test_true, preds_test, target_names=class_labels, output_dict=True)
    creport_df = pd.DataFrame(creport).transpose()
    creport_df.to_csv("classification_report_test.csv")
    print("Saved classification_report_test.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_val, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
    plt.title("Confusion Matrix - Validation")
    plt.tight_layout()
    plt.savefig("confusion_matrix_val.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
    plt.title("Confusion Matrix - Test")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png")
    plt.close()

    total_train_images = len(df_train)
    train_throughput = total_train_images / total_training_wallclock
    num_test_images = len(test_df_local)
    test_throughput = num_test_images / test_time

    print(f"Training throughput: {train_throughput:.2f} images/sec")
    print(f"Test throughput: {test_throughput:.2f} images/sec")

    baseline = all_train_times[0]
    speedups = [baseline / t for t in all_train_times]
    plt.figure()
    plt.plot(range(1, len(speedups)+1), speedups, marker="o")
    plt.title("MPI Speedup")
    plt.xlabel("Processes")
    plt.ylabel("Speedup")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speedup.png")
    plt.close()

    plt.figure()
    plt.plot(df_epochs["epoch"], df_epochs["loss"], marker="o")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    plt.savefig("loss_per_epoch.png")
    plt.close()

    plt.figure()
    plt.plot(df_epochs["epoch"], df_epochs["accuracy"], marker="o")
    plt.title("Training Accuracy per Epoch")
    plt.grid(True)
    plt.savefig("accuracy_per_epoch.png")
    plt.close()

    model_path = os.path.join(SAVE_DIR, "blood_cancer_model.h5")
    model.save(model_path)

    class_map = train_generator.class_indices
    inv_map = {int(v): str(k) for k, v in class_map.items()}
    with open(os.path.join(SAVE_DIR, "class_names.json"), "w") as f:
        json.dump(inv_map, f, indent=2)

    metrics = {
        "total_training_wallclock_s": total_training_wallclock,
        "val_time_s": val_time,
        "test_time_s": test_time,
        "train_throughput": train_throughput,
        "test_throughput": test_throughput,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "per_process_train_times_s": all_train_times
    }
    with open("metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open("evaluation_results.txt", "w") as f:
        f.write(f"Wall-clock: {total_training_wallclock:.2f}s\n")
        f.write(f"Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}\n")

comm.Barrier()
if rank == 0:
    print("\nDone (all ranks).")
