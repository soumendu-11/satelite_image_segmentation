import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from unet import UNet   # import from unet.py
import os

# --- Helper functions ---
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


def main():
    parser = argparse.ArgumentParser("UNet CAM Visualization")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory containing val.csv")
    parser.add_argument("--model_path", type=str, required=True, help="Trained UNet model path")
    parser.add_argument("--output_dir", type=str, default="cam_outputs", help="Directory to save CAM images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Device setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load model ---
    model = UNet(n_classes=6, in_channels=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- Register CAM hook on last conv ---
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.c9.double_conv[3].register_forward_hook(get_activation('final_conv'))

    # --- Load sample image + mask from val.csv ---
    val_df = pd.read_csv(f"{args.data_dir}/val.csv")
    image_path = val_df.iloc[1]['Image']
    mask_path = val_df.iloc[1]['Mask']

    patch_size = 256
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_crop = Image.fromarray(img_rgb).crop(
        (0, 0, (img.shape[1] // patch_size) * patch_size, (img.shape[0] // patch_size) * patch_size)
    )
    img_np = np.array(img_crop)

    img_norm = img_np / 255.0
    img_tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # --- Inference ---
    with torch.no_grad():
        output = model(img_tensor)
        predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # --- Load ground truth mask ---
    gt_mask = cv2.imread(mask_path)
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
    gt_crop = Image.fromarray(gt_mask).crop(
        (0, 0, (gt_mask.shape[1] // patch_size) * patch_size, (gt_mask.shape[0] // patch_size) * patch_size)
    )
    gt_mask_np = np.array(gt_crop)
    gt_label = rgb_to_2D_label(gt_mask_np)

    # --- Convert prediction and GT masks to RGB ---
    pred_rgb = label_to_rgb(predicted_mask)
    gt_rgb = label_to_rgb(gt_label)

    # --- Generate CAMs for all classes ---
    feature_maps = activation['final_conv']  # [1, 16, H, W]

    for class_idx in range(6):
        weights = model.out.weight[class_idx].squeeze()  # [16]
        cam = torch.einsum('bchw,c->bhw', feature_maps, weights)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        img_float = img_rgb.astype(np.float32) / 255
        overlayed = heatmap + img_float
        overlayed = overlayed / np.max(overlayed)

        # --- Save combined figure ---
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(img_rgb); axs[0].set_title("Original Image"); axs[0].axis('off')
        axs[1].imshow(pred_rgb); axs[1].set_title("Predicted Segmentation"); axs[1].axis('off')
        axs[2].imshow(gt_rgb); axs[2].set_title("Ground Truth Mask"); axs[2].axis('off')
        axs[3].imshow(overlayed); axs[3].set_title(f"CAM Overlay (Class {class_idx})"); axs[3].axis('off')

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"cam_class_{class_idx}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved CAM visualization for class {class_idx} at {save_path}")


if __name__ == "__main__":
    main()

