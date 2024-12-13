import os

def validating_model(file_path = r"D:\Image Segmentation (using YOLO)\models\yolov5\segment\val.py",
                     best_trained_model = r"D:\Image Segmentation (using YOLO)\models\yolov5\runs\train-seg\exp\weights\best.pt",
                     data_path = r"D:\Image Segmentation (using YOLO)\config\data.yaml",
                     img_size = 640):
    # Wrap paths containing spaces in double quotes
    command = f'python "{file_path}" --weights "{best_trained_model}" --data "{data_path}" --img {img_size}'
    os.system(command)

def testing_model(file_path = r"D:\Image Segmentation (using YOLO)\models\yolov5\segment\predict.py",
                  best_trained_model = r"D:\Image Segmentation (using YOLO)\models\yolov5\runs\train-seg\exp\weights\best.pt",
                  data_path = r"D:\Image Segmentation (using YOLO)\config\data.yaml",
                  img_size = 640,
                  test_path = r"D:\Image Segmentation (using YOLO)\data\images\test1",
                  project_path = r"D:\Image Segmentation (using YOLO)\output",
                  exp_name = "test1 (yolov5)",
                  conf_threshold = 0.5):
    # Wrap paths containing spaces in double quotes
    command = f'python "{file_path}" --weights "{best_trained_model}" --img {img_size} --conf {conf_threshold} --source "{test_path}" --project "{project_path}" --name "{exp_name}"'
    os.system(command)

if __name__ == "__main__":
    validating_model()

