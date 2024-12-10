import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AgeGenderDataset(Dataset):
    def __init__(self, root, train=True, transforms=None):
        """
        Args:
            root (str): Thư mục chứa dữ liệu
            train (bool): Nếu True thì dùng dữ liệu huấn luyện, nếu False thì dùng dữ liệu kiểm tra
            transforms (callable, optional): Các biến đổi cho dữ liệu
        """
        self.root = root
        self.train = train
        self.transforms = transforms
        self.data = []  # Danh sách các đường dẫn hình ảnh
        self.labels = []  # Danh sách nhãn của hình ảnh
        self._load_data()

    def _load_data(self):
        # Chọn thư mục train hoặc test
        phase = 'train' if self.train else 'test'
        data_dir = os.path.join(self.root, phase)

        # Duyệt qua các nhóm độ tuổi
        for age_group in os.listdir(data_dir):
            age_group_path = os.path.join(data_dir, age_group)
            if os.path.isdir(age_group_path):
                # Duyệt qua các giới tính
                for gender in os.listdir(age_group_path):
                    gender_path = os.path.join(age_group_path, gender)
                    if os.path.isdir(gender_path):
                        # Duyệt qua các hình ảnh
                        for img_name in os.listdir(gender_path):
                            img_path = os.path.join(gender_path, img_name)
                            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.data.append(img_path)
                                label = self._get_label(age_group, gender)
                                self.labels.append(label)

    def _get_label(self, age_group, gender):
        # Mã hóa nhãn cho độ tuổi và giới tính
        age_mapping = ['18-20', '21-30', '31-40', '41-50', '51-60']
        gender_mapping = ['Female', 'Male']

        age_label = age_mapping.index(age_group)
        gender_label = gender_mapping.index(gender)

        # Nhãn cuối cùng là sự kết hợp của độ tuổi và giới tính
        return age_label * 2 + gender_label  # 2 giới tính * 5 nhóm tuổi = 10 nhãn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        # Áp dụng các transform (resize, chuyển thành tensor, chuẩn hóa...)
        if self.transforms:
            image = self.transforms(image)

        return image, label


# Các transform cho training và validation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
