# %% init
'''import logger_config
logger = logger_config.get_logger(__name__)

import os

import gpu_selection
# 控制分配的GPU数量
num_gpus =1  # 你可以根据需要调整这个值
gpu_selection.select_gpus(num_gpus)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler


import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utilities3 import *
import glob
from skimage.metrics import structural_similarity as ssim

import networks

import my_tools
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from timeit import default_timer
import tqdm

from torch.optim import AdamW

from config import load_config
config = load_config()

# 创建训练和测试数据集
# train_dataset = my_tools.TIFDataset(root_dir=os.path.join(DATASET_PATH, 'training_data'), transform=my_tools.train_transform)
test_dataset = my_tools.TIFDataset(root_dir=os.path.join(config.DATASET_PATH, 'testing_data'))

# logger.debug("Train files: %d", len(train_dataset))
logger.debug("Valid files: %d", len(test_dataset))
# if len(train_dataset) == 0 or len(test_dataset) == 0:
#     logger.critical("No files found in the dataset")
#     raise ValueError("No files found in the dataset")

# 使用部分测试样本
test_subset_indices = list(range(len(test_dataset)))
test_subset = Subset(test_dataset, test_subset_indices)
test_subset_loader = DataLoader(test_subset, batch_size=config.batch_size_valid, shuffle=False, num_workers=0)

# train_sampler = get_train_sampler(train_dataset, ntrain_files)
# train_loader_with_sampler = my_tools.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)

# logger.info("Train files after limit: %d", len(train_loader_with_sampler)*batch_size)
logger.info("Valid files after limit: %d", len(test_subset_loader)*config.batch_size_valid)

sample_image, sample_class = test_dataset[0]
logger.debug("Sample image shape: %s", sample_image.shape)
logger.debug("Sample class: %s", sample_class)

path = 'eFIN'+"_step{}_".format(config.stage) + str(config.version) + '_ep'+str(config.epochs)+'_m' + str(config.modes) + '_w' + str(config.width)
path_model = '../Models/'+path

writer = SummaryWriter(os.path.join("../runs", path))

################################################################
# training and evaluation
################################################################

model = networks.FNO2d(config).cuda()
logger.info(f"FIN model size: {count_params(model)}")

optimizer = AdamW(list(model.parameters()), lr=config.learning_rate, weight_decay=1e-4, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

celoss = nn.CrossEntropyLoss()

# netD = Discriminator().cuda()
# criterionD = nn.BCELoss()
# optimizerD = torch.optim.SGD(netD.parameters(), lr=0.0001)

# perceptualloss = PerceptualLoss([0,1,2], [0.5,0.15,0.1], torch.device("cuda" if torch.cuda.is_available() else "cpu")).cuda()

max_valid_accuracy=0

if not os.path.exists(path_model):
    os.makedirs(path_model)

print_target = False
start_ep = -1
ep_relative = start_ep+1
if os.path.isfile(ckpt_path:=os.path.join(path_model,"ep_3792.pth")):
    logger.critical(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    pretrained_dict = {key.replace("module.", ""): value for key, value in checkpoint['model'].items()}
    model.load_state_dict(pretrained_dict)
else:
    logger.critical("No checkpoint found")
    raise RuntimeError("No checkpoint found")

###############valid##################
model.eval()
yy_list = []
zj_list = []
im_list = []
xx_list = [] # add this line
pred_list = []
confidence_list = []
correct_predictions_valid = 0
total_predictions_valid = 0
with torch.no_grad():
    for xx, yy in (pbar := tqdm.tqdm(test_subset_loader, dynamic_ncols=True)):
        loss = 0
        xx = xx.cuda()
        yy = yy.cuda()

        im = model(xx)

        # 计算准确率
        softmax_outputs = nn.Softmax(dim=1)(im)
        confidences, predicted = torch.max(softmax_outputs, 1)
        correct_predictions_valid += (predicted == yy).sum().item()
        total_predictions_valid += yy.size(0)

        yy_list.append(yy.detach().cpu().numpy())
        # zj_list.append(zj.detach().cpu().numpy())
        im_list.append(im.detach().cpu().numpy())
        # xx_list.append(xx.detach().cpu().numpy())
        pred_list.append(predicted.detach().cpu().numpy())
        confidence_list.append(confidences.detach().cpu().numpy())

yy = np.vstack(yy_list).reshape((-1,)+yy.shape[1:])
# zj = np.vstack(zj_list).reshape((-1,)+zj.shape[1:])
im = np.vstack(im_list).reshape((-1,)+im.shape[1:])
# xx = np.vstack(xx_list).reshape((-1,)+xx.shape[1:])
pred = np.vstack(pred_list).reshape((-1,))
confidence_list = np.vstack(confidence_list).reshape((-1,))

logger.critical(f"Valid accuracy: {correct_predictions_valid/total_predictions_valid*100:.2f}%")

# %% save im pred and confidence_list
np.savez(os.path.join('valid_results.npz'), im=im, pred=pred, confidence_list=confidence_list)
'''
# %% load im pred and confidence_list
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load(r"C:\Users\cmcbo\Downloads\valid_results.npz")
im = data['im']
pred = data['pred']
confidence_list = data['confidence_list']
yy = data['yy']


# %% 4 class confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
# 计算混淆矩阵
cm = confusion_matrix(yy, pred)
accuracy = accuracy_score(yy, pred)

# Set font and size
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 18
})

# 绘制混淆矩阵图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, '1+', '2+', '3+'], yticklabels=[0, '1+', '2+', '3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'Testing accuracy={accuracy*100:.1f}%')
plt.show()


# %% 2 class confusion matrix

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

merged_preds = merge_classes(pred)
merged_labels = merge_classes(yy)

# 计算混淆矩阵
cm = confusion_matrix(merged_labels, merged_preds)
accuracy = accuracy_score(merged_labels, merged_preds)

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r', xticklabels=['0 & 1+', '2+ & 3+'], yticklabels=['0 & 1+', '2+ & 3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'N=3, Testing accuracy={accuracy*100:.1f}%')
plt.show()

# %% N=3
# %% 4 class left out vs accuracy 

'''# 计算不同置信度阈值下的测试准确率和被排除数据点的百分比
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(yy)

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(pred, yy, confidence_list):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of Confidence threshold')
plt.subplots_adjust(top=0.9)
# plt.grid(True)
plt.show()'''

# Calculate testing accuracy and left out percentages for different confidence thresholds
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(yy)

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(pred, yy, confidence_list):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Find the CI value corresponding to leaving out 15% of the data
left_out_15_index = np.where(np.array(left_out_percentages) >= 15)[0][0]
ci_threshold = thresholds[left_out_15_index] * 100

# Plot the dashed grey lines
ax2.axhline(y=15, xmin=ci_threshold/100, xmax=1, color='grey', linestyle='--')
ax2.plot([ci_threshold, ci_threshold], [0, 15], color='grey', linestyle='--')

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of Confidence threshold')
plt.subplots_adjust(top=0.9)
plt.show()

print(f'The CI value corresponding to leaving out 15% of the data is approximately {ci_threshold:.2f}%')


# %% 2 class left out vs accuracy 
'''
# 计算不同置信度阈值下的测试准确率和被排除数据点的百分比
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(yy)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(pred, yy, confidence_list):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        filtered_preds = merge_classes(filtered_preds)
        filtered_labels = merge_classes(filtered_labels)
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of Confidence threshold')
# plt.grid(True)
plt.show()
'''

# Merge classes
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

# Calculate testing accuracy and left out percentages for different confidence thresholds
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(yy)

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(pred, yy, confidence_list):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        filtered_preds = merge_classes(filtered_preds)
        filtered_labels = merge_classes(filtered_labels)
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # If no data points meet the confidence threshold, mark as NaN
        left_out_percentages.append(100)

# Plot testing accuracy and left out percentages vs. confidence threshold
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Find the CI value corresponding to leaving out 15% of the data
left_out_15_index = np.where(np.array(left_out_percentages) >= 15)[0][0]
ci_threshold = thresholds[left_out_15_index] * 100

# Plot the dashed grey lines
ax2.axhline(y=15, xmin=ci_threshold/100, xmax=1, color='grey', linestyle='--')
ax2.plot([ci_threshold, ci_threshold], [0, 15], color='grey', linestyle='--')

fig.tight_layout()  # Adjust layout to prevent label overlap
plt.title('Effect of Confidence threshold')
plt.subplots_adjust(top=0.9)
plt.show()

print(f'The CI value corresponding to leaving out 15% of the data is approximately {ci_threshold:.2f}%')

# %% confusin matrix

# 给定的置信度阈值
# threshold = 0.61  # 4 class
threshold = 0.561  # 2 class

# 计算被排除数据点的比率和具体数量
filtered_preds = []
filtered_labels = []
total_samples = len(yy)

for pre, label, confidence in zip(pred, yy, confidence_list):
    if confidence >= threshold:
        filtered_preds.append(pre)
        filtered_labels.append(label)

left_out_count = total_samples - len(filtered_labels)
left_out_percentage = 100 * left_out_count / total_samples

print(f"Confidence threshold: {threshold*100:.1f}%")
print(f"Left out count: {left_out_count}")
print(f"Left out percentage: {left_out_percentage:.2f}%")

# 计算原始类别的混淆矩阵
cm_original = confusion_matrix(filtered_labels, filtered_preds)
accuracy_original = accuracy_score(filtered_labels, filtered_preds)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

merged_preds = merge_classes(filtered_preds)
merged_labels = merge_classes(filtered_labels)

# 计算合并类别的混淆矩阵
cm_merged = confusion_matrix(merged_labels, merged_preds)
accuracy_merged = accuracy_score(merged_labels, merged_preds)

# 打印混淆矩阵和准确率
print(f"Original Confusion Matrix:\n{cm_original}")
print(f"Original Accuracy: {accuracy_original*100:.2f}%")
print(f"Merged Confusion Matrix:\n{cm_merged}")
print(f"Merged Accuracy: {accuracy_merged*100:.2f}%")

# 绘制混淆矩阵图

sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', xticklabels=[0, '1+', '2+', '3+'], yticklabels=[0, '1+', '2+', '3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'N=3, Testing accuracy={accuracy_original*100:.1f}%, Left out={left_out_percentage:.1f}%')

plt.tight_layout()
plt.show()

sns.heatmap(cm_merged, annot=True, fmt='d', cmap='rocket_r', xticklabels=['0 & 1+', '2+ & 3+'], yticklabels=['0 & 1+', '2+ & 3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'N=3, Testing accuracy={accuracy_merged*100:.1f}%, Left out={left_out_percentage:.1f}%')

plt.tight_layout()
plt.show()


# %%  max CI

# 将结果划分为三段
num_samples = 284
first_scan_preds = pred[:num_samples]
second_scan_preds = pred[num_samples:2*num_samples]
third_scan_preds = pred[2*num_samples:]

first_scan_labels = yy[:num_samples]
second_scan_labels = yy[num_samples:2*num_samples]
third_scan_labels = yy[2*num_samples:]

first_scan_confidences = confidence_list[:num_samples]
second_scan_confidences = confidence_list[num_samples:2*num_samples]
third_scan_confidences = confidence_list[2*num_samples:]

# 对每个样本，找到三次扫描中置信度最大的预测
max_preds = []
max_labels = []
max_confidences = []

for i in range(num_samples):
    confidences = [first_scan_confidences[i], second_scan_confidences[i], third_scan_confidences[i]]
    preds = [first_scan_preds[i], second_scan_preds[i], third_scan_preds[i]]
    labels = [first_scan_labels[i], second_scan_labels[i], third_scan_labels[i]]

    max_confidence_index = np.argmax(confidences)
    max_confidences.append(confidences[max_confidence_index])
    max_preds.append(preds[max_confidence_index])
    max_labels.append(labels[max_confidence_index])

# %% 4 class left out vs accuracy 
'''# 计算不同置信度阈值下的测试准确率和被排除数据点的百分比
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(max_labels)

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(max_preds, max_labels, max_confidences):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of confidence threshold:\nHighest confidence method')
# plt.grid(True)
plt.subplots_adjust(top=0.85)
plt.show()'''

# 计算不同置信度阈值下的测试准确率和被排除数据点的百分比
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(max_labels)

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(max_preds, max_labels, max_confidences):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Find the CI value corresponding to leaving out 15% of the data
left_out_15_index = np.where(np.array(left_out_percentages) >= 15)[0][0]
ci_threshold = thresholds[left_out_15_index] * 100

# Plot the dashed grey lines
ax2.axhline(y=15, xmin=ci_threshold/100, xmax=1, color='grey', linestyle='--')
ax2.plot([ci_threshold, ci_threshold], [0, 15], color='grey', linestyle='--')

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of confidence threshold:\nHighest confidence method')
plt.subplots_adjust(top=0.85)
plt.show()

print(f'The CI value corresponding to leaving out 15% of the data is approximately {ci_threshold:.2f}%')


# %% 2 class left out vs accuracy 

'''# 计算不同置信度阈值下的测试准确率和被排除数据点的百分比
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(max_labels)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(max_preds, max_labels, max_confidences):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        filtered_preds = merge_classes(filtered_preds)
        filtered_labels = merge_classes(filtered_labels)
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of confidence threshold:\nHighest confidence method')
# plt.grid(True)
plt.subplots_adjust(top=0.85)
plt.show()'''

# 计算不同置信度阈值下的测试准确率和被排除数据点的百分比
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(max_labels)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(max_preds, max_labels, max_confidences):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        filtered_preds = merge_classes(filtered_preds)
        filtered_labels = merge_classes(filtered_labels)
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Find the CI value corresponding to leaving out 15% of the data
left_out_15_index = np.where(np.array(left_out_percentages) >= 15)[0][0]
ci_threshold = thresholds[left_out_15_index] * 100

# Plot the dashed grey lines
ax2.axhline(y=15, xmin=ci_threshold/100, xmax=1, color='grey', linestyle='--')
ax2.plot([ci_threshold, ci_threshold], [0, 15], color='grey', linestyle='--')

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of confidence threshold:\nHighest confidence method')
plt.subplots_adjust(top=0.85)
plt.show()

print(f'The CI value corresponding to leaving out 15% of the data is approximately {ci_threshold:.2f}%')


# %% confusin matrix

# 给定的置信度阈值
# threshold = 0.7367  # 4 class
threshold = 0.714  # 2 class

# 计算被排除数据点的比率和具体数量
filtered_preds = []
filtered_labels = []
total_samples = len(max_labels)

for pre, label, confidence in zip(max_preds, max_labels, max_confidences):
    if confidence >= threshold:
        filtered_preds.append(pre)
        filtered_labels.append(label)

left_out_count = total_samples - len(filtered_labels)
left_out_percentage = 100 * left_out_count / total_samples

print(f"Confidence threshold: {threshold*100:.1f}%")
print(f"Left out count: {left_out_count}")
print(f"Left out percentage: {left_out_percentage:.2f}%")

# 计算原始类别的混淆矩阵
cm_original = confusion_matrix(filtered_labels, filtered_preds)
accuracy_original = accuracy_score(filtered_labels, filtered_preds)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

merged_preds = merge_classes(filtered_preds)
merged_labels = merge_classes(filtered_labels)

# 计算合并类别的混淆矩阵
cm_merged = confusion_matrix(merged_labels, merged_preds)
accuracy_merged = accuracy_score(merged_labels, merged_preds)

# 打印混淆矩阵和准确率
print(f"Original Confusion Matrix:\n{cm_original}")
print(f"Original Accuracy: {accuracy_original*100:.2f}%")
print(f"Merged Confusion Matrix:\n{cm_merged}")
print(f"Merged Accuracy: {accuracy_merged*100:.2f}%")

# 绘制混淆矩阵图
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', xticklabels=[0, '1+', '2+', '3+'], yticklabels=[0, '1+', '2+', '3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'Max CI, Testing accuracy={accuracy_original*100:.1f}%, Left out={left_out_percentage:.1f}%')

plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.show()

# plt.subplot(1, 2, 2)
sns.heatmap(cm_merged, annot=True, fmt='d', cmap='rocket_r', xticklabels=['0 & 1+', '2+ & 3+'], yticklabels=['0 & 1+', '2+ & 3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'Max CI, Testing accuracy={accuracy_merged*100:.1f}%, Left out={left_out_percentage:.1f}%')

plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.show()

# %% ave CI

# 将结果划分为三段
num_samples = 284
first_scan_preds = pred[:num_samples]
second_scan_preds = pred[num_samples:2*num_samples]
third_scan_preds = pred[2*num_samples:]

first_scan_labels = yy[:num_samples]
second_scan_labels = yy[num_samples:2*num_samples]
third_scan_labels = yy[2*num_samples:]

# 对每个样本，计算三次扫描的预测类别的平均值，并四舍五入到最近的整数
avg_preds = []
avg_labels = []

for i in range(num_samples):
    preds = [first_scan_preds[i], second_scan_preds[i], third_scan_preds[i]]
    avg_pred = round(np.mean(preds))
    avg_preds.append(avg_pred)
    avg_labels.append(first_scan_labels[i])  # 假设所有扫描的标签是相同的

# 给定的置信度阈值
threshold = 0.5621  # 可以根据需要调整

# 计算被排除数据点的比率和具体数量
filtered_preds = []
filtered_labels = []
total_samples = len(avg_labels)

for pre, label, confidence in zip(avg_preds, avg_labels, confidence_list[:num_samples]):
    if confidence >= threshold:
        filtered_preds.append(pre)
        filtered_labels.append(label)

left_out_count = total_samples - len(filtered_labels)
left_out_percentage = 100 * left_out_count / total_samples

print(f"Confidence threshold: {threshold*100:.1f}%")
print(f"Left out count: {left_out_count}")
print(f"Left out percentage: {left_out_percentage:.2f}%")

# 计算原始类别的混淆矩阵
cm_original = confusion_matrix(filtered_labels, filtered_preds)
accuracy_original = accuracy_score(filtered_labels, filtered_preds)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

merged_preds = merge_classes(filtered_preds)
merged_labels = merge_classes(filtered_labels)

# 计算合并类别的混淆矩阵
cm_merged = confusion_matrix(merged_labels, merged_preds)
accuracy_merged = accuracy_score(merged_labels, merged_preds)

# 打印混淆矩阵和准确率
print(f"Original Confusion Matrix:\n{cm_original}")
print(f"Original Accuracy: {accuracy_original*100:.2f}%")
print(f"Merged Confusion Matrix:\n{cm_merged}")
print(f"Merged Accuracy: {accuracy_merged*100:.2f}%")

# 绘制混淆矩阵图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', xticklabels=[0, '1+', '2+', '3+'], yticklabels=[0, '1+', '2+', '3+'], cbar=False, square=True)
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'Ave, Testing accuracy={accuracy_original*100:.1f}%\nLeft out={left_out_percentage:.1f}%')

plt.subplot(1, 2, 2)
sns.heatmap(cm_merged, annot=True, fmt='d', cmap='rocket_r', xticklabels=['0 & 1+', '2+ & 3+'], yticklabels=['0 & 1+', '2+ & 3+'], cbar=False, square=True)
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'Ave, Testing accuracy={accuracy_merged*100:.1f}%\nLeft out={left_out_percentage:.1f}%')

plt.tight_layout()
plt.show()


# %% weighted ave CI 

# 将结果划分为三段
num_samples = 284
first_scan_preds = pred[:num_samples]
second_scan_preds = pred[num_samples:2*num_samples]
third_scan_preds = pred[2*num_samples:]

first_scan_labels = yy[:num_samples]
second_scan_labels = yy[num_samples:2*num_samples]
third_scan_labels = yy[2*num_samples:]

first_scan_confidences = confidence_list[:num_samples]
second_scan_confidences = confidence_list[num_samples:2*num_samples]
third_scan_confidences = confidence_list[2*num_samples:]

# 对每个样本，计算三次扫描的预测类别的平均值，并四舍五入到最近的整数
avg_preds = []
avg_labels = []
avg_confidences = []

for i in range(num_samples):
    preds = [first_scan_preds[i] * first_scan_confidences[i], second_scan_preds[i] * second_scan_confidences[i], third_scan_preds[i] * third_scan_confidences[i]]
    avg_pred = round(np.mean(preds)/np.mean([first_scan_confidences[i], second_scan_confidences[i], third_scan_confidences[i]]))

    avg_preds.append(avg_pred)
    avg_labels.append(first_scan_labels[i])  # 假设所有扫描的标签是相同的
    avg_confidences.append(np.mean([first_scan_confidences[i], second_scan_confidences[i], third_scan_confidences[i]]))

# %% 4 class left out vs accuracy 
# Calculate testing accuracy and left out percentages for different confidence thresholds
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(avg_labels)

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(avg_preds, avg_labels, avg_confidences):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # If no data points meet the confidence threshold, mark as NaN
        left_out_percentages.append(100)

# Plot testing accuracy and left out percentages vs. confidence threshold
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Find the CI value corresponding to leaving out 15% of the data
left_out_15_index = np.where(np.array(left_out_percentages) >= 15)[0][0]
ci_threshold = thresholds[left_out_15_index] * 100

# Plot the dashed grey lines
ax2.axhline(y=15, xmin=ci_threshold/100, xmax=1, color='grey', linestyle='--')
ax2.plot([ci_threshold, ci_threshold], [0, 15], color='grey', linestyle='--')

fig.tight_layout()  # Adjust layout to prevent label overlap
plt.title('Effect of Confidence threshold - Averaging method')
plt.subplots_adjust(top=0.9)
plt.show()

print(f'The CI value corresponding to leaving out 15% of the data is approximately {ci_threshold:.2f}%')

# %% 2 class left out vs accuracy 

'''# 计算不同置信度阈值下的测试准确率和被排除数据点的百分比
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(avg_labels)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(avg_preds, avg_labels, avg_confidences):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        filtered_preds = merge_classes(filtered_preds)
        filtered_labels = merge_classes(filtered_labels)
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # 如果没有数据点满足置信度阈值，则标记为NaN
        left_out_percentages.append(100)

# 绘制测试准确率和被排除数据点百分比随置信度阈值变化的曲线
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # 调整布局以防止标签重叠
plt.title('Effect of Confidence threshold - Averaging method')
# plt.grid(True)
plt.show()'''
# Merge classes
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

# Calculate testing accuracy and left out percentages for different confidence thresholds
thresholds = np.arange(0, 1.01, 0.01)
accuracies = []
left_out_percentages = []

total_samples = len(avg_labels)

for threshold in thresholds:
    filtered_preds = []
    filtered_labels = []
    for pre, label, confidence in zip(avg_preds, avg_labels, avg_confidences):
        if confidence >= threshold:
            filtered_preds.append(pre)
            filtered_labels.append(label)
    if len(filtered_labels) > 0:
        filtered_preds = merge_classes(filtered_preds)
        filtered_labels = merge_classes(filtered_labels)
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        accuracies.append(accuracy)
        left_out_percentage = 100 * (total_samples - len(filtered_labels)) / total_samples
        left_out_percentages.append(left_out_percentage)
    else:
        accuracies.append(np.nan)  # If no data points meet the confidence threshold, mark as NaN
        left_out_percentages.append(100)

# Plot testing accuracy and left out percentages vs. confidence threshold
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Confidence threshold (%)')
ax1.set_ylabel('Testing accuracy (%)', color=color)
ax1.plot(thresholds * 100, np.array(accuracies) * 100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
color = 'tab:orange'
ax2.set_ylabel('Left out (%)', color=color)
ax2.plot(thresholds * 100, left_out_percentages, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Find the CI value corresponding to leaving out 15% of the data
left_out_15_index = np.where(np.array(left_out_percentages) >= 15)[0][0]
ci_threshold = thresholds[left_out_15_index] * 100

# Plot the dashed grey lines
ax2.axhline(y=15, xmin=ci_threshold/100, xmax=1, color='grey', linestyle='--')
ax2.plot([ci_threshold, ci_threshold], [0, 15], color='grey', linestyle='--')

fig.tight_layout()  # Adjust layout to prevent label overlap
plt.title('Effect of Confidence threshold - Averaging method')
plt.subplots_adjust(top=0.9)
plt.show()

print(f'The CI value corresponding to leaving out 15% of the data for 2-class is approximately {ci_threshold:.2f}%')


# %% confusin matrix

threshold = 0.619  # 2 class

# 计算被排除数据点的比率和具体数量
filtered_preds = []
filtered_labels = []
total_samples = len(avg_labels)

for pre, label, confidence in zip(avg_preds, avg_labels, avg_confidences):
    if confidence >= threshold:
        filtered_preds.append(pre)
        filtered_labels.append(label)

left_out_count = total_samples - len(filtered_labels)
left_out_percentage = 100 * left_out_count / total_samples

print(f"Confidence threshold: {threshold*100:.1f}%")
print(f"Left out count: {left_out_count}")
print(f"Left out percentage: {left_out_percentage:.2f}%")

# 计算原始类别的混淆矩阵
cm_original = confusion_matrix(filtered_labels, filtered_preds)
accuracy_original = accuracy_score(filtered_labels, filtered_preds)

# 将类别进行合并
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

merged_preds = merge_classes(filtered_preds)
merged_labels = merge_classes(filtered_labels)

# 计算合并类别的混淆矩阵
cm_merged = confusion_matrix(merged_labels, merged_preds)
accuracy_merged = accuracy_score(merged_labels, merged_preds)

# 打印混淆矩阵和准确率
print(f"Original Confusion Matrix:\n{cm_original}")
print(f"Original Accuracy: {accuracy_original*100:.2f}%")
print(f"Merged Confusion Matrix:\n{cm_merged}")
print(f"Merged Accuracy: {accuracy_merged*100:.2f}%")

# 绘制混淆矩阵图

sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', xticklabels=[0, '1+', '2+', '3+'], yticklabels=[0, '1+', '2+', '3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'Ave, Testing accuracy={accuracy_original*100:.1f}%, Left out={left_out_percentage:.1f}%')

plt.tight_layout()
plt.show()

sns.heatmap(cm_merged, annot=True, fmt='d', cmap='rocket_r', xticklabels=['0 & 1+', '2+ & 3+'], yticklabels=['0 & 1+', '2+ & 3+'])
plt.xlabel('Predicted HER2 Score')
plt.ylabel('Actual HER2 Score')
plt.title(f'Ave, Testing accuracy={accuracy_merged*100:.1f}%, Left out={left_out_percentage:.1f}%')

plt.tight_layout()
plt.show()

# %% 
import numpy as np
from scipy import stats
import scipy.io as sio

# 假设我们有284个核心，每个核心有3次测量
num_cores = 284
measurements_per_core = 3

# 计算每个核心的预测一致性
core_consistency = {}
for core in range(num_cores):
    core_preds = []
    for measurement in range(measurements_per_core):
        index = core + measurement * num_cores
        core_preds.append(pred[index])
    
    # 计算模式（最常见的预测类别）
    mode = stats.mode(core_preds).mode
    
    # 计算一致性（与模式相同的预测比例）
    consistency = sum(p == mode for p in core_preds) / measurements_per_core
    core_consistency[core] = consistency

# 计算整体一致性
overall_consistency = sum(core_consistency.values()) / num_cores

print(f"Overall Consistency: {overall_consistency:.4f}")

# 可视化每个核心的一致性
plt.figure(figsize=(12, 6))
plt.bar(core_consistency.keys(), core_consistency.values())
plt.xlabel('Core')
plt.ylabel('Prediction Consistency')
plt.title('Prediction Consistency for each core')
plt.ylim(0, 1)  # 一致性的范围是0到1
plt.show()

# 计算一致性的统计信息
consistency_values = list(core_consistency.values())
consistency_mean = np.mean(consistency_values)
consistency_std = np.std(consistency_values)
consistency_median = np.median(consistency_values)

print(f"Consistency Mean: {consistency_mean:.4f}")
print(f"Consistency Standard Deviation: {consistency_std:.4f}")
print(f"Consistency Median: {consistency_median:.4f}")

# 绘制一致性的分布直方图
plt.figure(figsize=(10, 6))
plt.hist(consistency_values, bins=30, edgecolor='black')
plt.xlabel('Prediction Consistency')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Consistency values')
plt.xlim(0, 1)  # 一致性的范围是0到1
plt.show()

# Save data to .mat file
sio.savemat('consistency_data.mat', {
    'overall_consistency': overall_consistency,
    'core_consistency': core_consistency,
    'pred': pred
})

# %%ROC
# Set up the figure with a specific size
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the width and height as needed

# Function to merge classes
def merge_classes(labels):
    merged_labels = []
    for label in labels:
        if label == 0 or label == 1:
            merged_labels.append(0)
        elif label == 2 or label == 3:
            merged_labels.append(1)
    return merged_labels

# Calculate ROC curves for each case
def plot_roc_curve(preds, labels, confidences, title):
    merged_preds = merge_classes(preds)
    merged_labels = merge_classes(labels)

    fpr, tpr, _ = roc_curve(merged_labels, confidences)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f'{title}\n(AUC = {roc_auc:.2f})')

# Original case
plot_roc_curve(pred, yy, confidence_list, 'Original')

# Max CI case
plot_roc_curve(max_preds, max_labels, max_confidences, 'Max CI')

# Average CI case
plot_roc_curve(avg_preds, avg_labels, confidence_list[:num_samples], 'Average CI')

# Weighted Average CI case
plot_roc_curve(avg_preds, avg_labels, avg_confidences, 'Weighted\nAverage CI')

# Plot settings
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves for 2-Class Cases')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.subplots_adjust(right=0.7)
plt.show()
