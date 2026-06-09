"""Bead detection module — locates individual beads using YOLOv8n."""

class BeadDetector:
    """Detects individual beads on a perspective-corrected board image."""
    def __init__(self, model_path: str = "models/bead-best.pt", conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

    def _load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
