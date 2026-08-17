# Image Training Pipeline

A complete computer-vision pipeline: dataset → train → checkpoint → inference → evaluation.

1. `00_dataset.riko` — scan `datasets/raw` into a typed DATASET.
2. `01_train.riko` — trainable CNN (POST /api/train with `save_path=checkpoints/mnist.safetensors`).
3. `02_inference.riko` — load the checkpoint and run inference.
4. `03_evaluation.riko` — load the checkpoint for evaluation metrics.
