import os
import gdown

os.makedirs("checkpoints", exist_ok=True)
gdown.download(id="1sTm7sgaQnJE0vV7g5NOiKzUF5WGa7nzg", output="checkpoints/model_best.pth", quiet=False)
