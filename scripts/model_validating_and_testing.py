import os
from ultralytics import YOLO

def validate(model_path = r"D:\Image Segmentation (using YOLO)\weights\yolo11m_segmentation\weights\best.pt",
             data_path = r"D:\Image Segmentation (using YOLO)\config\data.yaml"):
    model = YOLO(model = model_path)
    model.val(data = data_path)

    return None

def testing(model_path = r"D:\Image Segmentation (using YOLO)\weights\yolo11m_segmentation\weights\best.pt",
            test_img_path = r"D:\Image Segmentation (using YOLO)\data\images\test1",
            output_path = r"D:\Image Segmentation (using YOLO)\output"):
    model = YOLO(model = model_path)

    os.makedirs(output_path, exist_ok = True)

    results = model.predict(
        source = test_img_path,
        conf = 0.5,
        save = True,
        project = output_path,
        name = 'test1',
        imgsz = 640
    )

    print(f"Predicted images saved to {output_path}")

    return None





if __name__ == "__main__":
    testing()