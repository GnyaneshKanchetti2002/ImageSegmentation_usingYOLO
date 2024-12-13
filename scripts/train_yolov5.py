import os

def train_yolov5(train_file = r"D:\Image Segmentation (using YOLO)\models\yolov5\segment\train.py",
                 data_yaml=r"D:\Image Segmentation (using YOLO)\config\data.yaml",
                 model=r"D:\Image Segmentation (using YOLO)\models\yolov5s-seg.pt",
                 epochs=100,
                 batch_size=1,
                 img_size=640):
    command = f'python "{train_file}" --img "{img_size}" --batch {batch_size} --epochs {epochs} --data "{data_yaml}" --weights "{model}" --cache'
    os.system(command)


if __name__ == "__main__":
    train_yolov5()