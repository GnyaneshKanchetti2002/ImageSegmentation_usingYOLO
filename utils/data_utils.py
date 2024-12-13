import numpy as np
import cv2
import json

def get_masks(train_img_path=r"D:\Image Segmentation (using YOLO)\data\images\train\\",
              ann_path=r"D:\Image Segmentation (using YOLO)\data\annotations\annotations(VGG).json"):
    with open(ann_path, 'r') as file:
        vgg_data = json.load(file)

    images = []
    masks = []

    class_mapping = {'Face': 1, 'Eye': 2, 'Nose': 3, 'lip': 4, 'Ear': 5, 'Eyebrow': 6, 'Glasses': 7}
    k = 0

    for image_id, image_data in vgg_data.items():
        img = cv2.imread(train_img_path + image_id)
        img_resize = cv2.resize(img, (640, 640))
        image = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        images.append(image / 255)

        mask = np.zeros((640, 640), dtype=np.uint8)

        for region in image_data['regions'].values():
            all_x_points = []
            all_y_points = []

            x_points = region['shape_attributes']['all_points_x']
            y_points = region['shape_attributes']['all_points_y']

            for i in range(len(x_points)):
                all_x_points.append(x_points[i] * (640 / img.shape[1]))
                all_y_points.append(y_points[i] * (640 / img.shape[0]))

            class_name = region['region_attributes']['label']

            class_id = class_mapping[class_name]

            polygon = np.array(list(zip(all_x_points, all_y_points)), dtype=np.int32)
            polygon_mask = np.zeros((640, 640), dtype=np.uint8)
            polygon_mask = cv2.fillPoly(polygon_mask, [polygon], 1)

            mask[(polygon_mask > 0) & ((mask == 0) | (mask < class_id)) & (mask != 2) & (mask != 6)] = class_id

        print(k, np.unique(mask))
        k += 1

        masks.append(mask)

    return images, masks