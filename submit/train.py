import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision
from torchvision import transforms
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FasterRCNN
from transforms import PILToTensor, Compose, FixedSizeCrop
from presets import DetectionPresetTrain
from presets import DetectionPresetEval
from utils import collate_fn
from coco_utils import get_coco
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch,evaluate

#Paths
DATA_PATH = "images/"
TRAIN_DATA_PATH = "images/train2017"
VAL_DATA_PATH = "images/test2017"
TRAIN_ANN_PATH = (
    "annotations/instances_train2017.json"
)
VAL_ANN_PATH = (
    "annotations/instances_val2017.json"
)

#Hyperparameters
OUTPUT_DIR = "./output"
DEVICE = "cpu" #or cuda
BATCH_SIZE = 8
EPOCHS = 20
WORKERS = 0
LEARNING_RATE = 1e-2
MOMENTUM = 0.9


def main():
    #Create output directory if doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


    device = torch.device(DEVICE)

    # Image preprocessing: 
        # random vertical/horizontal flip
        # jittering: random small shifts
    train_preprocessing = DetectionPresetTrain(data_augmentation="lsj")
    val_preprocessing = DetectionPresetEval()
    train_dataset = get_coco(DATA_PATH, "train", transforms=train_preprocessing)
    test_dataset = get_coco(DATA_PATH, "val", transforms=val_preprocessing)

    #Data loading
    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        collate_fn=collate_fn,
    )

    valloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=WORKERS
    )

    #Define the model backbone and anchor generator

    backbone = torchvision.models.mobilenet_v2().features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone,
        rpn_anchor_generator=anchor_generator,
        num_classes=91, #91 classes for COCO dataset
    )

    model.to(device)

    #optimizer setup
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE, momentum=0.9)

    #training loop
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, labels in trainloader:
            # train_one_epoch(model,optimizer,trainloader,DEVICE,epoch,1)

            images = list(image.to(device) for image in images)
            labels = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in labels
            ]

            optimizer.zero_grad()
            loss_dict = model(images, labels)
            losses = sum(loss for loss in loss_dict.values())
            losses_reduced = sum(loss for loss in loss_dict.values())
            loss_value = losses_reduced.item()
            losses.backward()
            epoch_loss += loss_value
            print(loss_value)
            optimizer.step()

        print(epoch_loss)
    

    

if __name__ == "__main__":
    main()
