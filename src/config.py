
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_VIDEOS_DIR = PROJECT_ROOT / "Dataset_Videos"
DATASET_NUMPY_DIR = PROJECT_ROOT / "Dataset_numpy"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Sequence configuration
NUM_FRAMES = 40

# Landmark configuration
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33
COORDINATES_PER_LANDMARK = 3

NUM_FEATURES = (
    2 * NUM_HAND_LANDMARKS * COORDINATES_PER_LANDMARK
    + NUM_POSE_LANDMARKS * COORDINATES_PER_LANDMARK
)

# Classification configuration
NUM_CLASSES = 12
CONFIDENCE_THRESHOLD = 0.8

# Supported image extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
