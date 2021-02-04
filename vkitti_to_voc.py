import os
from collections import defaultdict 
from glob import glob
import cv2
from xml.etree.ElementTree import Element, SubElement, ElementTree

path = 'data/vkitti/'
labels = ['Car', 'Pedestrian', 'Person', 'Van', 'Truck', 'Tram', 'Cyclist']
mapping = {
    'Car': 'car', 'Pedestrian': 'person', 'Person': 'person', 
    'Van': 'Van', 'Truck': 'Truck', 'Tram': 'Tram', 'Cyclist': 'Cyclist'
}

def write_xml(annotations, idx, shape, xml_path):
    root = Element('annotation')
    SubElement(root, 'filename').text = str(idx).zfill(6) + '.xml'
    SubElement(root, 'segmented').text = '0'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(shape[1])
    SubElement(size, 'height').text = str(shape[0])
    SubElement(size, 'depth').text = '3'

    for ann in annotations:
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = ann['label']
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = ann['truncated']
        SubElement(obj, 'difficult').text = '0'
        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = ann['xmin']
        SubElement(bbox, 'ymin').text = ann['ymin']
        SubElement(bbox, 'xmax').text = ann['xmax']
        SubElement(bbox, 'ymax').text = ann['ymax']

    tree = ElementTree(root)
    tree.write(xml_path)


def read_annotation_text(text_path):
    annotations = defaultdict(list)
    with open(text_path, 'r') as f:
        for idx, l in enumerate(f.readlines()):
            if idx == 0:
                continue
            l = l.split()
            frame, tid, label, truncated, occluded, alpha, l, t, r, b = l[:10]
            if label in labels:
                label = mapping[label]
                annotations[int(frame)].append({
                    'label': label,
                    'truncated': truncated,
                    'occluded': occluded,
                    'xmin': l, 'xmax': r,
                    'ymin': t, 'ymax': b
                })
    return annotations


def main():
    os.makedirs(os.path.join(path, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(path, 'ImageSets/Main'), exist_ok=True)
    os.makedirs(os.path.join(path, 'JPEGImages'), exist_ok=True)
    numbers = glob(os.path.join(path, 'vkitti_1.3.1_rgb/*'))
    idx = 0
    for num in numbers:
        basename = os.path.basename(num)
        for file in glob(os.path.join(num, '*')):
            a = os.path.basename(file)
            text_path = os.path.join(path, 'vkitti_1.3.1_motgt', f'{basename}_{a}.txt')
            annotations = read_annotation_text(text_path)
            for img in glob(os.path.join(file, '*.png')):
                frame_num = int(os.path.basename(img)[:-4])
                img_array = cv2.imread(img)
                cv2.imwrite(os.path.join(path, 'JPEGImages', f'{str(idx).zfill(6)}.jpg'), img_array)
                write_xml(
                    annotations[frame_num], idx, img_array.shape,
                    os.path.join(path, 'Annotations', f'{str(idx).zfill(6)}.xml')
                )
                idx += 1
                print('write ', )
    
    with open(os.path.join(path, 'ImageSets/Main', 'train.txt'), 'w') as f:
        for i in range(idx):
            f.write(f'{i}\n')


if __name__ == '__main__':
    main()