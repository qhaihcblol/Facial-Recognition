import torch
from src.face_detection.config.retinaface_config import get_config
from src.face_detection.training.retinaface_training import main

class Args:
    train_data = "./src/face_detection/data/widerface/widerface/train"
    network = "mobilenet_v2"
    num_workers = 0  # nên để nhỏ để tránh lỗi RAM
    num_classes = 2
    batch_size = 8
    print_freq = 10
    learning_rate = 1e-3
    lr_warmup_epochs = 1
    power = 0.9
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.1
    save_dir = "./src/face_detection/weights"
    resume = True


args = Args()
cfg = get_config(args.network)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Define device in the global scope
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # (tùy chọn, giúp an toàn hơn trên Windows)
    main(args, cfg)
