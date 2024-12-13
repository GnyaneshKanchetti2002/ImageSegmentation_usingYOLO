from ultralytics import YOLO

def train_model():
    model = YOLO(r"D:\Image Segmentation (using YOLO)\models\yolo11m-seg.pt")
    results = model.train(
        data = r"D:\Image Segmentation (using YOLO)\config\data.yaml",
        epochs = 100,
        imgsz = 640,
        batch = 1,
        flipud = 0.25,  # Vertical flip probability
        fliplr = 0.25,  # Horizontal flip probability
        degrees = 10,  # Random rotation degrees
        scale = 0.25,  # Scale range, e.g., 0.5 means [0.5, 1.5] of original size
        project="D:/Image Segmentation (using YOLO)/weights",
        name="yolo11m_segmentation"
    )


if __name__ == "__main__":
    train_model()