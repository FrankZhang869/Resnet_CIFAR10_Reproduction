import json
import matplotlib.pyplot as plt

# Load metrics
with open("plain_metrics.json", "r") as f:
    metrics = json.load(f)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -------------------------
# Validation Accuracy
# -------------------------
for model_name, data in metrics.items():
    axes[0].plot(
        (data["epochs"])[:200],
        (data["val_acc"])[:200],
        label=model_name
    )

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(75, 95)
axes[0].grid(True)
axes[0].legend()

# -------------------------
# Validation Loss
# -------------------------
for model_name, data in metrics.items():
    axes[1].plot(
        (data["epochs"])[:200],
        (data["val_loss"])[:200],
        label=model_name
    )

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(True)
axes[1].legend()

# -------------------------
# Training Loss
# -------------------------
for model_name, data in metrics.items():
    axes[2].plot(
        (data["epochs"])[:200],
        (data["train_loss"])[:200],
        label=model_name
    )

axes[2].set_title("Training Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].set_xlim(0, 200)
axes[2].set_ylim(0, 0.4)
axes[2].grid(True)
axes[2].legend()

# Overall figure title
fig.suptitle(
    "Plain Network Training on CIFAR-10",
    fontsize=16
)

plt.tight_layout()
plt.show()

"""

import json
import matplotlib.pyplot as plt

# Load metrics
with open("resnet_metrics.json", "r") as f:
    metrics = json.load(f)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -------------------------
# Validation Accuracy
# -------------------------
for model_name, data in metrics.items():
    axes[0].plot(
        data["epochs"],
        data["val_acc"],
        label=model_name
    )

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(80, 95)
axes[0].grid(True)
axes[0].legend()

# -------------------------
# Validation Loss
# -------------------------
for model_name, data in metrics.items():
    axes[1].plot(
        data["epochs"],
        data["val_loss"],
        label=model_name
    )

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_ylim(80, 95)
axes[1].grid(True)
axes[1].legend()

# -------------------------
# Training Loss
# -------------------------
for model_name, data in metrics.items():
    axes[2].plot(
        data["epochs"],
        data["train_loss"],
        label=model_name
    )

axes[2].set_title("Training Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].set_xlim(50, 200)
axes[2].set_ylim(0, 0.25)
axes[2].grid(True)
axes[2].legend()

fig.suptitle(
    "ResNet Training on CIFAR-10",
    fontsize=16
)

plt.tight_layout()

plt.show()

"""