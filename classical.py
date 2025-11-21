import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report

# ------------------- PATHS -------------------
DATASET_PATH = r"C:\Users\HP\Desktop\HPC_Project\Blood cell Cancer [ALL]"
SAVE_DIR     = r"C:\Users\HP\Desktop\HPC_Project\classical_model"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE  = 32
TARGET_SIZE = (224, 224)
EPOCHS      = 20

# ------------------- SCAN IMAGES -------------------
print("Scanning dataset...")
classes_dirs = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

data = []
for cls in classes_dirs:
    cls_path = os.path.join(DATASET_PATH, cls)
    for fname in os.listdir(cls_path):
        fp = os.path.join(cls_path, fname)
        try:
            with Image.open(fp) as im: im.verify()
            data.append([fp, cls])
        except:
            print(f"Skipping corrupted file: {fp}")

df = pd.DataFrame(data, columns=["filepath", "class"]).sample(frac=1, random_state=42)
print("Total valid images:", len(df))
print(df["class"].value_counts())

# ------------------- TRAIN/VAL/TEST SPLIT -------------------
n = len(df)
df_train = df.iloc[:int(0.7*n)]
df_val   = df.iloc[int(0.7*n):int(0.85*n)]
df_test  = df.iloc[int(0.85*n):]

# ------------------- DATA GENERATORS -------------------
train_gen = ImageDataGenerator(
    rescale=1/255., rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.1, horizontal_flip=True, brightness_range=[0.8,1.2],
    fill_mode='nearest'
).flow_from_dataframe(
    df_train, x_col="filepath", y_col="class",
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True
)

val_gen = ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
    df_val, x_col="filepath", y_col="class",
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

test_gen = ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
    df_test, x_col="filepath", y_col="class",
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)

# ------------------- MODEL -------------------
model = Sequential([
    Input(shape=(224,224,3)),
    Conv2D(32, 3, activation='relu'), MaxPooling2D(2),
    Conv2D(64, 3, activation='relu'), MaxPooling2D(2),
    Conv2D(128, 3, activation='relu'), MaxPooling2D(2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes_dirs), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------- TRAINING LOOP -------------------
history = {"loss": [], "acc": []}
epoch_times = []
baseline_time = None

print("\nTraining started...\n")

for epoch in range(EPOCHS):
    ep_start = time.time()
    ep_loss, ep_acc = [], []
    
    pbar = tqdm(total=len(train_gen), unit="batch", desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for X, y in train_gen:
        loss, acc = model.train_on_batch(X, y)
        ep_loss.append(loss); ep_acc.append(acc)
        pbar.set_postfix({"loss": f"{np.mean(ep_loss):.4f}", "acc": f"{np.mean(ep_acc):.4f}"})
        pbar.update(1)
        if pbar.n >= len(train_gen): break
    pbar.close()

    ep_time = time.time() - ep_start
    epoch_times.append(ep_time)
    if baseline_time is None: baseline_time = ep_time
    speedup = baseline_time / ep_time
    throughput = len(df_train) / ep_time

    history["loss"].append(np.mean(ep_loss))
    history["acc"].append(np.mean(ep_acc))

    print(f"Epoch {epoch+1}: Time={ep_time:.2f}s | Loss={history['loss'][-1]:.4f} | "
          f"Acc={history['acc'][-1]:.4f} | Throughput={throughput:.2f} img/s | Speedup={speedup:.2f}x")

train_time = sum(epoch_times)

# ------------------- VALIDATION & TEST -------------------
val_start = time.time(); val_loss, val_acc = model.evaluate(val_gen, verbose=1); val_time = time.time() - val_start
test_start = time.time(); test_loss, test_acc = model.evaluate(test_gen, verbose=1); test_time = time.time() - test_start

# ------------------- PREDICTIONS & METRICS -------------------
preds_val  = np.argmax(model.predict(val_gen), axis=1)
preds_test = np.argmax(model.predict(test_gen), axis=1)
y_val, y_test = val_gen.classes, test_gen.classes
labels = list(train_gen.class_indices.keys())

cm_val  = confusion_matrix(y_val, preds_val)
cm_test = confusion_matrix(y_test, preds_test)

cls_report = classification_report(y_test, preds_test, target_names=labels)

# ------------------- SAVE RESULTS -------------------
model.save(os.path.join(SAVE_DIR, "blood_cancer_model.h5"))
with open(os.path.join(SAVE_DIR, "class_names.json"), "w") as f:
    json.dump({v:k for k,v in train_gen.class_indices.items()}, f, indent=2)

metrics = {
    "train_time": train_time, "val_time": val_time, "test_time": test_time,
    "train_throughput": float(len(df_train)/train_time),
    "test_throughput": float(len(df_test)/test_time),
    "val_acc": float(val_acc), "test_acc": float(test_acc),
    "val_loss": float(val_loss), "test_loss": float(test_loss)
}
with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f: json.dump(metrics, f, indent=4)

pd.DataFrame({"epoch": range(1,EPOCHS+1),"time":epoch_times,
              "loss":history["loss"],"acc":history["acc"]}).to_csv(os.path.join(SAVE_DIR, "epoch_times.csv"), index=False)

with open(os.path.join(SAVE_DIR, "classification_report_test.txt"), "w") as f: f.write(cls_report)

# ------------------- PLOTS -------------------
plt.figure(); plt.plot(history["loss"]); plt.title("Loss per Epoch"); plt.grid()
plt.savefig(os.path.join(SAVE_DIR, "loss.png")); plt.close()

plt.figure(); plt.plot(history["acc"]); plt.title("Accuracy per Epoch"); plt.grid()
plt.savefig(os.path.join(SAVE_DIR, "accuracy.png")); plt.close()

plt.figure(figsize=(7,5))
sns.heatmap(cm_val, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Validation Confusion Matrix")
plt.savefig(os.path.join(SAVE_DIR, "cm_val.png")); plt.close()

plt.figure(figsize=(7,5))
sns.heatmap(cm_test, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title("Test Confusion Matrix")
plt.savefig(os.path.join(SAVE_DIR, "cm_test.png")); plt.close()

print(f"\n Training Complete! Files saved to:\n{SAVE_DIR}")
