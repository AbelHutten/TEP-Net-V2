import matplotlib.pyplot as plt
import pandas as pd

# Matplotlib setting
plt.rcParams["axes.autolimit_mode"] = "round_numbers"

# Data loading
df = pd.read_csv("/home/abel/Documents/tepnet_v2/TEP-Net-V2/data/eval.csv")
df = df[df["runtime"] == "tensorrt"]
df = df.sort_values(by="latency")

# Color and label position offset
color_map = {
    "classification": (214 / 256, 39 / 256, 40 / 256),
    "regression": (44 / 256, 160 / 256, 44 / 256),
    "segmentation": (31 / 256, 119 / 256, 180 / 256),
}

x_offset = -0.034
y_offset_upper = 0.00038
y_offset_lower = -0.00076
label_offset_map = {
    "classification": [False, True, False, True, True, False, True],
    "regression": [False, True, False, True, True, False, True],
    "segmentation": [False, True, True, False, True, True, True],
}

# Plotting
fig, ax = plt.subplots()

for method in ["classification", "regression", "segmentation"]:
    classification = df[df["method"] == method]
    classification_latency = list(classification["latency"])
    classification_iou = list(classification["iou"])
    classification_labels = [
        bb[0].upper() + "N" + bb[-2:].upper() for bb in classification["backbone"]
    ]

    plt.plot(
        classification_latency,
        classification_iou,
        "o-",
        color=color_map[method],
        label=method,
    )

    for i, label in enumerate(classification_labels):
        y_offset = y_offset_upper if label_offset_map[method][i] else y_offset_lower
        plt.text(
            classification_latency[i] + x_offset,
            classification_iou[i] + y_offset,
            label,
            fontsize=9,
            fontweight="bold",
            color=color_map[method],
            fontfamily="monospace",
        )

plt.xlabel("Latency TensorRT AMP (ms/img)")
plt.ylabel("Accuracy (IoU)")
plt.xlim([0.2, 1.4])
plt.ylim([0.96, 0.98])
plt.legend()
plt.grid()
plt.show()
