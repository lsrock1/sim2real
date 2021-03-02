from mmdet.apis import init_detector, inference_detector
import mmcv
from glob import glob


# Specify the path to model config and checkpoint file
config_file = 'configs/vital/faster_rcnn_r50_fpn_1x_airplane.py'
checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_airplane/epoch_12.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
images = sorted(glob('data/airplane/detection6/*.png'))[:50]

# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
for img in images:
    result = inference_detector(model, img)
    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file=f'airplanes/{img}')