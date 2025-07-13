import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def denormalize(image_tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return image_tensor * std + mean

def overlay_heatmap_on_image(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # [H, W, 3]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    heatmap = heatmap.detach().cpu().to(torch.float32).numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def visualize_explanations(
    images, heatmaps, labels, class_ids_list,
    mean, std,
    save=True,
    show=True,
    show_original=True,
    save_path="./figures",
    index_names=None,
):
    if save:
        os.makedirs(save_path, exist_ok=True)

    N = len(images)

    for i in range(N):
        image = denormalize(images[i], mean, std)
        label = int(labels[i])
        class_ids = class_ids_list[i]

        overlays = []
        titles = []

        for j in range(3):
            class_id = class_ids[j]
            heatmap_tensor = heatmaps[i][j]
            heatmap = heatmap_tensor.mean(dim=0)

            overlay = overlay_heatmap_on_image(image, heatmap)
            overlays.append(overlay)
            name = index_names[j] if index_names else f"target{j}"
            titles.append(f"Class: {class_id} ({name})")

            if save:
                out_path = os.path.join(save_path, f"img{i}_cls{class_id}_{name}.png")
                cv2.imwrite(out_path, overlay)

        if show:
            if show_original:
                fig, axs = plt.subplots(2, 2, figsize=(8, 6))

                # Position 0,0: Original image
                axs[0, 0].imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
                axs[0, 0].set_title(f"Original (GT: {label})")
                axs[0, 0].axis("off")

                # Position 0,1; 1,0; 1,1: Overlays
                positions = [(0, 1), (1, 0), (1, 1)]
                for j, (row, col) in enumerate(positions):
                    axs[row, col].imshow(overlays[j][..., ::-1])  # BGR -> RGB
                    axs[row, col].set_title(titles[j])
                    axs[row, col].axis("off")

            else:
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                for j in range(3):
                    axs[j].imshow(overlays[j][..., ::-1])
                    axs[j].set_title(titles[j])
                    axs[j].axis("off")

            plt.tight_layout()
            plt.show()

