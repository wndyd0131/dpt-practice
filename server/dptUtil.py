# This file contains code adapted from the DPT repository:
# https://github.com/intel-isl/DPT
# Licensed under the MIT License.
# Copyright (c) 2021 Intel Corporation

from io import BytesIO

import torch.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dpt.models import DPTDepthModel

class DPTModelRunner:
    def __init__(self):
        self.model = DPTDepthModel(
            path=None,
            non_negative=True,
            scale=1.0,
            shift=0.0,
            invert=False
        )
        self.device = torch.device('cpu')

    def predict(self, x):
        # preprocessing
        image = Image.open(BytesIO(x)).convert("RGB")
        width, height = image.size

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        image = transform(image).unsqueeze(0)

        model = self.model.to(self.device)

        checkpoint = torch.load("../checkpoint/dpt_checkpoint_epoch.pth", map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            output = model(image)
            # postprocessing
            if output.dim() == 3:
                output = output.unsqueeze(1)
            output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)
            pred = output.squeeze(1)
            pred = torch.clamp(pred, min=1e-3, max=10.0)
        return pred

    def createDepthFile(self, pred):
        depth_map = pred.numpy()  # convert from tensor to numpy to use image processing libraries
        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map = (depth_map * 255).astype(np.uint8)

        print(f"Shape {depth_map.shape}")
        print(f"Squeezed {depth_map.squeeze(0)}")
        depth_map = depth_map.squeeze(0)
        # depth_image = Image.fromarray(depth_map)
        # depth_image.save('./1_depth.jpg')

        # prediction = prediction.squeeze()


        fig, ax = plt.subplots()
        cax = ax.imshow(depth_map, cmap='plasma')
        fig.colorbar(cax, ax=ax)
        ax.set_title("Depth Map")
        ax.axis('off')

        buf_plot = BytesIO()
        plt.savefig(buf_plot, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        #
        # plt.imshow(depth_map, cmap='plasma')
        # plt.title("Depth Map")
        # plt.colorbar()
        # plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)

        fig2, ax2 = plt.subplots(figsize=(depth_map.shape[1] / 100, depth_map.shape[0] / 100), dpi=100)
        ax2.imshow(depth_map, cmap='plasma')
        ax2.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf_depth = BytesIO()
        plt.savefig(buf_depth, format='png', dpi=100, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close(fig2)

        buf_plot.seek(0)
        buf_depth.seek(0)

        return buf_plot, buf_depth

if __name__ == '__main__':
    with open('./1.jpg', 'rb') as f1:
        image = f1.read()

        dptModelRunner = DPTModelRunner()

        prediction = dptModelRunner.predict(image)

        depth_map = prediction.numpy() # convert from tensor to numpy to use image processing libraries
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        # depth_map = (depth_map * 255).astype(np.uint8)

        print(f"Shape {depth_map.shape}")
        print(f"Squeezed {depth_map.squeeze(0)}")
        depth_map = depth_map.squeeze(0)
        # depth_image = Image.fromarray(depth_map)
        # depth_image.save('./1_depth.jpg')

        prediction = prediction.squeeze()
        plt.imshow(prediction, cmap='plasma')
        plt.title("Depth Map")
        plt.colorbar()
        plt.savefig(f"./result/depth.png")