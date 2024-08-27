

import monai
import logging
import os
import sys

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report
import argparse
import torch.nn.functional as nnf

import monai
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    Orientation,
    Rotate90,
    EnsureChannelFirst,
    Compose,
    ScaleIntensityRangePercentiles,
    ResizeWithPadOrCrop
)

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cuda device: ",torch.cuda.current_device())
print(device)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str, default='./data', help="data path")
parser.add_argument("--path_checkpoint", type=str, default='./pretrained_model', help="path to checkpoint")
parser.add_argument("--path_result", type=str, default='./', help="path to save results")
parser.add_argument("--model", type=str, default='DenseNet121', help="classification model")
args = parser.parse_args()

os.makedirs(args.path_result, exist_ok=True)

########################################################################
#### Set data loader for testing ##############
########################################################################
print('##### Setting test data #####')
images_test = []
labels_test = []
images_path = args.path_data

test_classes = ['T1w_pre', 'T1w_art', 'T1w_ven', 'T1w_del', 'T2w', 'T2fs', 'DWI', 'ADC'] 
for item in test_classes:
    vol_type_path = os.path.join(images_path, item)
    vol_type_files = sorted(os.listdir(vol_type_path))

    for ifileName in vol_type_files:
        images_test.append(os.path.join(vol_type_path, ifileName))
        labels_test.append(item)

# map target values to an integer, one-hot encode
labels_test = pd.Series(labels_test)
labels_test = labels_test.map({'T1w_pre': 0, 'T1w_art': 1, 'T1w_ven': 2, 'T1w_del': 3, 'T2w': 4, 'T2fs': 5, 'DWI': 6, 'ADC': 7}).to_numpy()
labels_test = torch.nn.functional.one_hot(torch.as_tensor(labels_test)).float()
labels_test_np = labels_test.cpu().numpy()
labels_test_df = pd.DataFrame(labels_test_np)
labels_test_df.to_csv(os.path.join(args.path_result, 'labels_test_df.csv'))

test_transforms = Compose([
    ScaleIntensityRangePercentiles(1, 99, 0, 1, clip=True, relative=False), 
    EnsureChannelFirst(), 
    ResizeWithPadOrCrop((256, 256, 36), "symmetric", "constant"),
    Orientation(axcodes='LAS'),
    Rotate90(k=3, spatial_axes=(0, 1))
])
test_ds = ImageDataset(image_files=images_test, labels=labels_test, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=pin_memory)
print('-------- Total # of test data: %d' % len(labels_test))

########################################################################
#### Set model ##############
########################################################################
print('##### Setting 3D model: %s #####' % args.model)
if args.model == 'DenseNet121':
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=8).to(device)
else:
    raise NameError('Wrong network name.')


########################################################################
#### Test Models ##############
########################################################################
all_probs = []
for fold in range(1,6):
    print('##### Test MRI_classifier (Fold %d) #####' % fold)
    checkpoint_dir = os.path.join(args.path_checkpoint, 'best_metric_model_fold_%s.pth'%fold)
    state_dict = torch.load(checkpoint_dir, map_location='cuda:0')
    model.load_state_dict(state_dict)
    model.eval()

    step_test = 0
    all_probabilities = torch.tensor([], dtype=torch.float32, device=device)
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    for test_data in test_loader:
        step_test += 1
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        with torch.no_grad():
            test_outputs = model(test_images)
            y_pred = torch.cat([y_pred, test_outputs], dim=0)
            batch_probabilities = nnf.softmax(test_outputs, dim=1)
            all_probabilities = torch.cat([all_probabilities, batch_probabilities], 0)
            predictions = float(test_outputs.argmax(dim=1)[0].cpu().detach().numpy())
            actuals = float(test_labels.argmax(dim=1)[0].cpu().detach().numpy())
        print('---%05d | label: %.1f | predict: %.1f' % (step_test, actuals, predictions))

    all_probabilities_np = all_probabilities.cpu().numpy()
    all_probs.append(all_probabilities_np)
   

########################################################################
#### Ensemble the probabilities from 5-fold models ##############
########################################################################
print('##### Results from ensembling the 5-fold models #####')
ensemble_probs = (all_probs[0] + all_probs[1] + all_probs[2] + all_probs[3] + all_probs[4])/5
ensemble_probs_df = pd.DataFrame(ensemble_probs).transpose()
ensemble_probs_df.to_csv(os.path.join(args.path_result,'ensemble_probs.csv'))

predictions = np.argmax(ensemble_probs, axis=1).tolist()
actuals = np.argmax(labels_test_np, axis=1).tolist()

labels = ['T1w-p', 'T1w-a', 'T1w-v', 'T1w-d', 'T2w', 'T2fs', 'DWI', 'ADC']
confusion_matrix_result = confusion_matrix(actuals, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7])
ax = sns.heatmap(confusion_matrix_result, annot= True, xticklabels = labels, yticklabels = labels, cmap='Blues', fmt='g')
ax.set(xlabel='Predicted', ylabel='Actual')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
plt.title("Test Outcomes: Actual vs. Predicted")
plt.savefig(os.path.join(args.path_result,"ensemble_confusion_matrix.pdf"), format='pdf')

classification_report = classification_report(actuals, predictions, target_names=labels, digits=4, output_dict=True)
classification_report_df = pd.DataFrame(classification_report).transpose()
classification_report_df.to_csv(os.path.join(args.path_result, 'ensemble_classification_report.csv'))
print(classification_report_df)
print("Ensemble AUC: ",metrics.roc_auc_score(y_true=labels_test_np, y_score=ensemble_probs, average='weighted'))
