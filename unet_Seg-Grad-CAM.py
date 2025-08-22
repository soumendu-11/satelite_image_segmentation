import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from unet import UNet   # import from unet.py

# -----------------------------
# Color utilities (unchanged)
# -----------------------------
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return np.array(tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4)))

COLOR_MAP = {
    0: hex_to_rgb('#3C1098'),  # Building
    1: hex_to_rgb('#8429F6'),  # Land
    2: hex_to_rgb('#6EC1E4'),  # Road
    3: hex_to_rgb('#FEDD3A'),  # Vegetation
    4: hex_to_rgb('#E2A929'),  # Water
    5: hex_to_rgb('#9B9B9B')   # Unlabeled
}

def rgb_to_2D_label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    for k, v in COLOR_MAP.items():
        matches = np.all(label == v, axis=-1)
        label_seg[matches] = k
    return label_seg

def label_to_rgb(label_2d):
    rgb_img = np.zeros((label_2d.shape[0], label_2d.shape[1], 3), dtype=np.uint8)
    for k, v in COLOR_MAP.items():
        rgb_img[label_2d == k] = v
    return rgb_img

# -----------------------------
# Seg-Grad-CAM helper
# -----------------------------
class SegGradCAM:
    """
    Seg-Grad-CAM for semantic segmentation (per-class heatmaps).
    target_layer: a convolutional layer whose activations have spatial dimensions (B, C, H, W).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            # Save activations (feature maps)
            self.activations = out

        def bwd_hook(module, grad_in, grad_out):
            # Save gradients wrt the output of target_layer
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_backward_hook(bwd_hook)

    @torch.no_grad()
    def _normalize(self, cam):
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min) < 1e-8:
            return torch.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)

    def generate(self, logits, class_idx, roi_mask=None):
        """
        logits: [B, C, H, W] raw (pre-softmax) outputs from the model (requires grad on the path!)
        class_idx: integer class id to explain
        roi_mask: [B, 1, H, W] float mask in {0,1} to weight the class score spatially (optional).
                  If None or empty, fallback to mean over the whole map.
        Returns: CAM upsampled to input size of logits [B, 1, H, W] in [0,1]
        """
        assert logits.requires_grad, "Logits must require grad for Seg-Grad-CAM."

        B, C, H, W = logits.shape
        class_map = logits[:, class_idx:class_idx+1, :, :]  # [B,1,H,W]

        if roi_mask is not None and roi_mask.sum() > 0:
            score = (class_map * roi_mask).sum() / (roi_mask.sum() + 1e-8)
        else:
            # Fallback: mean score over the entire class map
            score = class_map.mean()

        # Backprop to get gradients at target layer
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        # Gradients: [B, C_feat, H_feat, W_feat]
        grads = self.gradients
        feats = self.activations

        # Channel weights (global average pooling of gradients across spatial dims)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C_feat, 1, 1]

        # Weighted sum over channels (ReLU)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))  # [B,1,H_feat,W_feat]

        # Upsample CAM to logits size
        cam_up = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)

        # Normalize to [0,1]
        cam_up = self._normalize(cam_up)
        return cam_up  # [B,1,H,W]

# -----------------------------
# Main script
# -----------------------------
def main():
    parser = argparse.ArgumentParser("UNet Seg-Grad-CAM Visualization")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory containing val.csv")
    parser.add_argument("--model_path", type=str, required=True, help="Trained UNet model path")
    parser.add_argument("--output_dir", type=str, default="seg_gradcam_outputs", help="Directory to save CAM images")
    parser.add_argument("--use_gt_roi", action="store_true",
                        help="If set, build ROI from ground-truth class pixels instead of predicted mask")
    parser.add_argument("--index", type=int, default=1, help="Row index from val.csv to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load model ---
    model = UNet(n_classes=6, in_channels=3).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Choose target layer for Seg-Grad-CAM ---
    # Prefer the last conv in the final decoder block (high-resolution semantics).
    # Adjust these attributes to match your UNet definition if needed.
    try:
        target_layer = model.c9.double_conv[3]  # common: Conv2d/ReLU sequence; index 3 is often the second Conv2d or ReLU
    except Exception:
        # Fallback: try last module in the block
        target_layer = list(model.c9.double_conv.children())[-1]
    print(f"Target layer for Seg-Grad-CAM: {target_layer.__class__.__name__}")

    seg_cam = SegGradCAM(model, target_layer)

    # --- Load sample image + mask from val.csv ---
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    image_path = val_df.iloc[args.index]['Image']
    mask_path = val_df.iloc[args.index]['Mask']

    patch_size = 256
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Crop to patch-aligned region (optional, matches your original code)
    img_crop = Image.fromarray(img_rgb).crop(
        (0, 0,
         (img_rgb.shape[1] // patch_size) * patch_size,
         (img_rgb.shape[0] // patch_size) * patch_size)
    )
    img_np = np.array(img_crop).astype(np.float32) / 255.0  # HWC, [0,1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    img_tensor.requires_grad_(False)

    # --- Forward pass WITH gradients enabled for logits path ---
    img_tensor_for_grad = img_tensor.clone().detach().requires_grad_(True)
    logits = model(img_tensor_for_grad)  # [1,C,H,W], pre-softmax
    pred_mask = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]  # [H,W]

    # --- Load ground truth mask (for optional ROI) ---
    gt_bgr = cv2.imread(mask_path)
    if gt_bgr is None:
        raise FileNotFoundError(f"Could not read mask at {mask_path}")
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    gt_crop = Image.fromarray(gt_rgb).crop(
        (0, 0,
         (gt_rgb.shape[1] // patch_size) * patch_size,
         (gt_rgb.shape[0] // patch_size) * patch_size)
    )
    gt_mask_np = np.array(gt_crop)
    gt_label = rgb_to_2D_label(gt_mask_np)  # [H,W]

    # --- Pretty RGB masks for visualization ---
    pred_rgb = label_to_rgb(pred_mask)
    gt_rgb_vis = label_to_rgb(gt_label)

    # --- Generate Seg-Grad-CAMs for all classes ---
    H, W = img_np.shape[:2]
    for class_idx in range(6):
        # Build ROI from predicted or GT pixels belonging to class_idx
        if args.use_gt_roi:
            roi_bool = (gt_label == class_idx)
        else:
            roi_bool = (pred_mask == class_idx)

        roi_mask = torch.from_numpy(roi_bool.astype(np.float32))[None, None, :, :].to(device)  # [1,1,H,W]

        # Compute Seg-Grad-CAM (will backprop through model)
        cam = seg_cam.generate(logits, class_idx, roi_mask=roi_mask)  # [1,1,H,W] in [0,1]
        cam_np = cam.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H,W]

        # Convert CAM to heatmap and overlay on original image
        cam_resized = cam_np  # already [H,W] matching logits
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET).astype(np.float32) / 255.0
        img_float = (img_np).astype(np.float32)  # [H,W,3] in [0,1]
        overlayed = heatmap + img_float
        overlayed = overlayed / np.maximum(overlayed.max(), 1e-8)

        # --- Save combined figure ---
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(img_np); axs[0].set_title("Original Image"); axs[0].axis('off')
        axs[1].imshow(pred_rgb); axs[1].set_title("Predicted Segmentation"); axs[1].axis('off')
        axs[2].imshow(gt_rgb_vis); axs[2].set_title("Ground Truth Mask"); axs[2].axis('off')
        axs[3].imshow(overlayed); axs[3].set_title(f"Seg-Grad-CAM (Class {class_idx})"); axs[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"seg_gradcam_class_{class_idx}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved Seg-Grad-CAM for class {class_idx} at {save_path}")

if __name__ == "__main__":
    main()
