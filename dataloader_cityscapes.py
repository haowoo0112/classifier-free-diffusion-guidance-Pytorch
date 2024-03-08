from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = os.listdir(os.path.join(folder_path, "images"))
        self.label_files = os.listdir(os.path.join(folder_path, "labels"))
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([64, 64]),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.transform_label = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([64, 64]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, "images", self.image_files[idx])
        img = Image.open(img_name)
        label_name = os.path.join(self.folder_path, "labels", self.label_files[idx])
        label = Image.open(label_name)

        img = self.transform_image(img)
        label = self.transform_label(label)
        # 可以進行進一步的前處理，例如轉換成 NumPy 數組，正規化等
        label = (label * 255).to(torch.int64)
        return label, img

def showImage():
    # 資料夾路徑，根據實際情況修改
    folder_path = 'cityscapes/train'
    dataset = CustomDataset(folder_path)

    # 使用 DataLoader 進行批次載入
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 使用dataloader進行迭代
    for inputs, ouputs in dataloader:
        inputs_np = np.array(inputs)
        outputs_np = np.array(ouputs)
        # inputs_np = np.transpose(inputs_np, (1, 2, 0))
        # outputs_np = np.transpose(outputs_np, (1, 2, 0))
        # 設置子圖
        fig, axs = plt.subplots(4, 2, figsize=(10, 15))

        for i in range(4):
            # 顯示 input
            axs[i, 0].imshow(np.transpose(inputs_np[i], (1, 2, 0)))
            axs[i, 0].set_title('Input')

            # 顯示 output
            axs[i, 1].imshow(np.transpose(outputs_np[i], (1, 2, 0)))
            axs[i, 1].set_title('Output')

        plt.show()
        break  # 只顯示一個批次的圖像，可以根據需要調整

# folder_path = "cityscapes/train"
# image_files = os.listdir(folder_path)
# img_name = os.path.join(folder_path, image_files[0])
# img = Image.open(img_name)
# width, height = img.size
# output_img = img.crop((0, 0, width // 2, height))
# input_img = img.crop((width // 2, 0, width, height))
# print(np.array(input_img))