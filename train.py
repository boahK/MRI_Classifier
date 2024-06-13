import monai
import logging
import os
import sys
import argparse
import math
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import csv
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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path_data", type=str, default='./Preprocessing/resampled_volumes/', help="path to dataset")
parser.add_argument("--path_checkpoint", type=str, default='./checkpoint', help="path to save checkpoint model")
parser.add_argument("--model", type=str, default='DenseNet121', help="classification model")
parser.add_argument("--gpu_ids", type=int, default=0, help="index of the gpu cards")
parser.add_argument("--valid_rate", type=int, default=0.2, help="Ratio of the # of validation to the # of training data")
parser.add_argument("--max_epochs", type=int, default=25, help="The number of training epochs")
args = parser.parse_args()

gpu_ids = args.gpu_ids
device = torch.device("cuda:%d"%gpu_ids if torch.cuda.is_available() else "cpu")
print("Cuda device: ", torch.cuda.current_device())

labels_annon = {'T1w_pre': 0, 'T1w_art': 1, 'T1w_ven': 2, 'T1w_del': 3, 'T2w': 4, 'T2fs': 5, 'DWI': 6, 'ADC': 7}
train_transforms = Compose([
    ScaleIntensityRangePercentiles(1, 99, 0, 1, clip=True, relative=False), 
    EnsureChannelFirst(), 
    ResizeWithPadOrCrop((256, 256, 36), "symmetric", "constant"),
    Orientation(axcodes='LAS'),
    Rotate90(k=3, spatial_axes=(0, 1))
])

val_transforms = Compose([
    ScaleIntensityRangePercentiles(1, 99, 0, 1, clip=True, relative=False), 
    EnsureChannelFirst(), 
    ResizeWithPadOrCrop((256, 256, 36), "symmetric", "constant"),
    Orientation(axcodes='LAS'),
    Rotate90(k=3,spatial_axes=(0, 1))
])
for fold in range(1,6):
    ########################################################################
    #### Set data loader for training and validation ##############
    ######################################################################## 
    print('##### Setting training & validation data (Fold %d) #####' %fold)
    images_train = []
    images_valid = []
    labels_train = []
    labels_valid = []

    images_path = args.path_data
    mri_classes = ['T1w_pre', 'T1w_art', 'T1w_ven', 'T1w_del', 'T2w', 'T2fs', 'DWI', 'ADC'] 
    for item in mri_classes:
        vol_type_path = os.path.join(images_path, item)
        vol_type_files = sorted(os.listdir(vol_type_path))

        iclass_totNum = len(vol_type_files)
        valid_totNum = int(iclass_totNum*args.valid_rate)
        fold_valid_data = vol_type_files[(fold-1)*valid_totNum:fold*valid_totNum]
        fold_train_data = []
        for ifileName in vol_type_files:
            if not ifileName in fold_valid_data:
                fold_train_data.append(ifileName)
        for ifileName in fold_train_data:
            images_train.append(os.path.join(vol_type_path, ifileName))
            labels_train.append(item)
        for ifileName in fold_valid_data:
            images_valid.append(os.path.join(vol_type_path, ifileName))
            labels_valid.append(item)

    # map target values to an integer, one-hot encode
    labels_train = pd.Series(labels_train)
    labels_train = labels_train.map(labels_annon).to_numpy().astype(np.int64)
    labels_train = torch.nn.functional.one_hot(torch.as_tensor(labels_train)).float()

    # map target values to an integer, one-hot encode
    labels_valid = pd.Series(labels_valid)
    labels_valid = labels_valid.map(labels_annon).to_numpy().astype(np.int64)
    labels_valid = torch.nn.functional.one_hot(torch.as_tensor(labels_valid)).float()

    # create a training data loader
    train_ds = ImageDataset(image_files=images_train, labels=labels_train, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=pin_memory)
    print('-------- Total # of training data: %d' % len(labels_train))

    # create a validation data loader
    val_ds = ImageDataset(image_files=images_valid, labels=labels_valid, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=pin_memory)
    print('-------- Total # of validation data: %d' % len(labels_valid))

    ########################################################################
    #### Set model ##############
    ########################################################################
    print('##### Setting 3D model: %s #####' % args.model)
    if args.model == 'DenseNet121':
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=8).to(device)
    else:
        raise NameError('Wrong network name.')

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    ########################################################################
    #### Training Model ##############
    ########################################################################
    print('##### Training the model (Fold %d) #####' %fold)
    best_metric = -1
    best_metric_epoch = -1
    writer = SummaryWriter()
    max_epochs = args.max_epochs

    output_dir = os.path.join(args.path_checkpoint, "%s_3d_classifier" %args.model)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "training_validation_metrics_fold_%d.csv"%fold), 'w') as m:
        writer1 = csv.writer(m)
        writer1.writerow(["epoch", "training_loss", "validation_loss", "training_accuracy", "validation_accuracy"])

        for epoch in range(max_epochs):
            print("----------")
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0

            num_correct_train = 0
            metric_count_train = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                value_train = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1))
                metric_count_train += len(value_train)
                num_correct_train += value_train.sum().item()

            metric_train = num_correct_train / metric_count_train
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            model.eval()

            num_correct = 0.0
            metric_count = 0
            epoch_loss_valid = 0
            step_valid = 0
            best_validation_loss = 10
            for val_data in val_loader:
                step_valid += 1
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                with torch.no_grad():
                    val_outputs = model(val_images)
                    value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                    loss_valid = loss_function(val_outputs, val_labels)
                    epoch_loss_valid += loss_valid.item()
                    metric_count += len(value)
                    num_correct += value.sum().item()

            epoch_loss_valid /= step_valid
            metric = num_correct / metric_count

            if epoch_loss_valid < best_validation_loss:
                best_validation_loss = epoch_loss_valid
                torch.save(model.state_dict(),  os.path.join(output_dir,"lowest_validation_loss_model_fold_%d.pth" %fold))
                print("saved new lowest validation loss model")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_dir, "best_metric_model_fold_%d.pth" %fold))
                print("saved new best validation metric model")

            print(f"Current epoch: {epoch + 1} current accuracy: {metric:.4f} ")
            print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
            writer.add_scalar("val_accuracy", metric, epoch + 1)
            writer1.writerow([epoch + 1, epoch_loss, epoch_loss_valid, metric_train, metric])

    print("Fold %d Training completed, best_metric: %.4f at epoch %d" %(fold, best_metric, best_metric_epoch))
    writer.close()
