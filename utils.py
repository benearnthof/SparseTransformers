import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def get_batch(split, cfg, device_type="cuda"):
    if split == 'train':
        data = np.memmap(os.path.join("/root/data_dir", 'train.bin'), dtype=np.uint8, mode='r')
    else:
        data = np.memmap(os.path.join("/root/data_dir", 'val.bin'), dtype=np.uint8, mode='r')
    
    if not cfg.overfit:
        # overfitting on one image to test the architecture
        ix = torch.randint(len(data)//cfg.block_size - cfg.block_size, (cfg.batch_size,))
    else:
        ix = torch.zeros(cfg.batch_size, dtype=int)
    x_list, y_list = [], []
    for i in ix:
        # i = tensor(4634)
        offset = i * cfg.block_size
        x_seq = data[offset:offset+cfg.block_size].astype(np.int64)
        # y is the same as y, offset by one extra pixel
        y_seq = data[offset+1:offset+1+cfg.block_size].astype(np.int64)
        # patch last byte
        y_seq[-1] = x_seq[-1]  # repeat the last token instead of leaking
        x_list.append(torch.from_numpy(x_seq))
        y_list.append(torch.from_numpy(y_seq))
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    if device_type == 'cuda':
        x, y = x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(cfg.device, non_blocking=True)
    else:
        x, y = x.to(cfg.device), y.to(cfg.device)
    return x, y

# reshape for visualization
def to_img(t1d):
    arr = t1d.detach().cpu().numpy().astype(np.uint8)
    img = arr.reshape(3, 32, 32).transpose(1, 2, 0)  # HWC
    return img

def save_cifar_image(tensor_3072: torch.Tensor, filename: str):
    """
    Convert a 3072-long tensor of bytes (flattened CIFAR-10 image)
    into a 32x32 RGB image and save to disk.
    """
    assert tensor_3072.numel() == 3072, "Input must be length 3072"
    # move to CPU and numpy
    image = to_img(tensor_3072)
    pil_img = Image.fromarray(image)
    pil_img.save(filename)
    print(f"Saved image to {filename}")

# To visualize single channels for debugging we need to vstack two separate tensors
# since numpy reshapes by row 
# arr = cur.detach().cpu().numpy()
# # First half (512 values) → reshape into 16×32 (top half)
# top_half = arr[:512].reshape(16, 32)
# # Second half (512 values, masked) → reshape into 16×32 (bottom half)
# bottom_half = arr[512:].reshape(16, 32)
# # Stack vertically → 32×32 image
# temp = np.vstack([top_half, bottom_half])
# # Save as grayscale image
# img = Image.fromarray(temp.astype(np.uint8), mode="L")
# img.save("first_channel_predicted.jpg")

# TODO: Generalize this for Imagenet 64
def generate_samples(model, cfg, n=4, temperature=1.0, top_k=None, save_path=None, split="val"):
    """
    Shows n rows of triplets:
      [masked input | model prediction | ground truth]
    Bottom half of each channel is masked in the input.
    Prediction is produced by autoregressively filling only the masked parts,
    keeping unmasked tokens fixed.
    This should be correct for the flattened cifar tensors since the first 
    1024 bytes represent the first color channel. So during training the model 
    sees only one color channel for sequences <= 1024 and only the first two color
    channels for sequences <= 2048.
    split argument can be set to "train" for debugging purposes
    """
    model.eval()
    with torch.no_grad():
        X, _ = get_batch(split=split, cfg=cfg)
        X = X[:n]  # (n, 3072)
        device = X.device
        img_size = 32
        num_pixels = img_size * img_size      # 1024 per channel
        half_pixels = (img_size // 2) * img_size  # 512 per channel
        # mask inputs
        X_masked = X.clone()
        for c in range(3):
            start = c * num_pixels + half_pixels
            end   = (c + 1) * num_pixels
            X_masked[:, start:end] = 0
        # save_cifar_image(X_masked[0], "masked.jpg")
        # fill masked regions per channel
        preds = []
        for i in range(n):
            # 512 entries for the first color channel
            cur = X[i, :half_pixels]
            # Predict the masked first color channel
            cur = model.generate(cur.unsqueeze(0), max_new_tokens=half_pixels,
                                 temperature=temperature, top_k=top_k)[0]  # now 1024
            # Predict the masked entries in the second channel
            cur = torch.cat([cur, X[i, 1024:1024+half_pixels]], dim=0)      # now 1536
            cur = model.generate(cur.unsqueeze(0), max_new_tokens=half_pixels,
                                 temperature=temperature, top_k=top_k)[0]  # now 2048
            # Predict the masked third channel
            cur = torch.cat([cur, X[i, 2048:2048+half_pixels]], dim=0)      # now 2560
            cur = model.generate(cur.unsqueeze(0), max_new_tokens=half_pixels,
                                 temperature=temperature, top_k=top_k)[0]  # now 3072

            assert cur.numel() == 3072
            preds.append(cur)

        preds = torch.stack(preds).to(device)  # (n, 3072)

        fig, axes = plt.subplots(n, 4, figsize=(12, 4*n))
        if n == 1:
            axes = np.array([axes])  # normalize shape to [1,4]
        for i in range(n):
            pred_img = to_img(preds[i])
            target_img = to_img(X[i])
            diff_img = np.abs(pred_img.astype(np.int16) - target_img.astype(np.int16)).astype(np.uint8)

            axes[i, 0].imshow(to_img(X_masked[i])); axes[i, 0].set_title("Input (masked)"); axes[i, 0].axis("off")
            axes[i, 1].imshow(pred_img);            axes[i, 1].set_title("Prediction");     axes[i, 1].axis("off")
            axes[i, 2].imshow(target_img);          axes[i, 2].set_title("Target");         axes[i, 2].axis("off")
            axes[i, 3].imshow(diff_img); axes[i, 3].set_title("Diff (abs)");   axes[i, 3].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved eval images to {save_path}")
        else:
            plt.show()

# TODO: merge both into single class
@torch.no_grad()
def generate_samples_pipe(model_engine, cfg, n=4, temperature=1.0, top_k=None, save_path=None, split="val"):
    """
    Same as above but should work with pipeline parallel GPTPipe
    GPTPipe inherits from PipelineModule which implements .forward implicitly 
    https://github.com/deepspeedai/DeepSpeed/blob/80033a82938f6cd8ce4988a63c914941e7a8f324/deepspeed/runtime/pipe/module.py#L340
    """
    raw_model = model_engine.module  # unwrap the GPTPipe
    raw_model.eval()

    # Load a small batch from disk
    # TODO: pass in data loader to generalize for Imagenet64
    X, _ = get_batch(split=split, cfg=cfg)
    X = X[:n].to(model_engine.device)  # (n, 3072)

    device = model_engine.device
    img_size = 32
    num_pixels = img_size * img_size      # 1024 per channel
    half_pixels = (img_size // 2) * img_size  # 512 per channel

    # mask inputs
    X_masked = X.clone()
    for c in range(3):
        start = c * num_pixels + half_pixels
        end   = (c + 1) * num_pixels
        X_masked[:, start:end] = 0

    preds = []
    for i in range(n):
        cur = X[i, :half_pixels]
        # autoregressive filling channel by channel
        cur = raw_model.generate(cur.unsqueeze(0), max_new_tokens=half_pixels,
                                 temperature=temperature, top_k=top_k)[0]
        cur = torch.cat([cur, X[i, 1024:1024+half_pixels]], dim=0)
        cur = raw_model.generate(cur.unsqueeze(0), max_new_tokens=half_pixels,
                                 temperature=temperature, top_k=top_k)[0]
        cur = torch.cat([cur, X[i, 2048:2048+half_pixels]], dim=0)
        cur = raw_model.generate(cur.unsqueeze(0), max_new_tokens=half_pixels,
                                 temperature=temperature, top_k=top_k)[0]

        assert cur.numel() == 3072
        preds.append(cur)

    preds = torch.stack(preds).to(device)

    fig, axes = plt.subplots(n, 4, figsize=(12, 4*n))
    if n == 1:
        axes = np.array([axes])
    for i in range(n):
        pred_img = to_img(preds[i])
        target_img = to_img(X[i])
        diff_img = np.abs(pred_img.astype(np.int16) - target_img.astype(np.int16)).astype(np.uint8)

        axes[i, 0].imshow(to_img(X_masked[i])); axes[i, 0].set_title("Input (masked)"); axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_img);            axes[i, 1].set_title("Prediction");     axes[i, 1].axis("off")
        axes[i, 2].imshow(target_img);          axes[i, 2].set_title("Target");         axes[i, 2].axis("off")
        axes[i, 3].imshow(diff_img);            axes[i, 3].set_title("Diff (abs)");     axes[i, 3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved eval images to {save_path}")
    else:
        plt.show()


def get_flops():
    # peak fp16 flops for GPUs we're using (for mfu calculation)
    PEAK_FLOPS = {
        # dense referene values, with sparsity they are ~double
        "NVIDIA RTX A4000": 19.1e12,
        "NVIDIA RTX A5000": 27.8e12, 
        "NVIDIA RTX 5090": 104e12,
        "NVIDIA A100": 312e12,
        "NVIDIA H100": 989e12,
        "NVIDIA H200": 989e12,
        "NVIDIA B200": 2250e12,
    }
    gpu_name = torch.cuda.get_device_name(0)
    for key in PEAK_FLOPS:
        if key in gpu_name:
            return PEAK_FLOPS[key]
    return None  # unknown GPU


