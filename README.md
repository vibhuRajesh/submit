# Skyline Nav AI Coding Challenge

The model used was the Faster R CNN: and it depends on the following libraries:

os,

torch,

torch.utils.data: DataLoader,

torchvision.datasets: CocoDetection,


torchvision,


torchvision: transforms,


torchvision.models.detection,


torchvision.models.detection.faster_rcnn: FasterRCNN,


transforms: PILToTensor, Compose, FixedSizeCrop,


presets: DetectionPresetTrain,


presets: DetectionPresetEval,


utils: collate_fn,


coco_utils: get_coco,


torchvision.models.detection.rpn: AnchorGenerator,


engine: train_one_epoch,evaluate
 

The training loop shows the training process and the consequent decrease in the model's loss in each epoch and for each sample.

The limitations of this model are that it doesn't provide the bounding boxes/counting on the output image, 
