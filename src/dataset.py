import os
import gzip
import numpy as np
import torch
from urllib.request import urlretrieve


class FashionMNISTDataset:
    """
    FashionMNIST dataset class
    Args:
        root: root directory of the dataset
        train: if True, use the training set, otherwise use the test set
        transform: transform to apply to the data
    """
    def __init__(self, root="./data", train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        self.train_data_file = os.path.join(self.root, "train-images-idx3-ubyte.gz")
        self.train_labels_file = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
        self.test_data_file = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
        self.test_labels_file = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")

        self.download_data()

        # load the dataset
        if self.train:
            self.data = self.load_images(self.train_data_file)
            self.targets = self.load_labels(self.train_labels_file)
        else:
            self.data = self.load_images(self.test_data_file)
            self.targets = self.load_labels(self.test_labels_file)

    def download_data(self):
        """
        Download the dataset from the internet
        """
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

        files = {
            "train_data": ("train-images-idx3-ubyte.gz", self.train_data_file),
            "train_labels": ("train-labels-idx1-ubyte.gz", self.train_labels_file),
            "test_data": ("t10k-images-idx3-ubyte.gz", self.test_data_file),
            "test_labels": ("t10k-labels-idx1-ubyte.gz", self.test_labels_file),
        }

        for _, (url_file, local_file) in files.items():
            if not os.path.exists(local_file):
                print(f"Downloading {url_file}...")
                urlretrieve(base_url + url_file, local_file)

    def load_images(self, file_path):
        """
        Load the images from the file
        Args:
            file_path: path to the file
        Returns:
            images: numpy array of images
        """
        with gzip.open(file_path, "rb") as f:
            # remove the header (16 bytes)
            images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
            return images

    def load_labels(self, file_path):
        """
        Load the labels from the file
        Args:
            file_path: path to the file
        Returns:
            labels: numpy array of labels
        """
        with gzip.open(file_path, "rb") as f:
            # remove the header (8 bytes)
            labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            return labels

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the item at the given index
        Args:
            idx: index of the item
        Returns:
            img: image
            target: label
        """
        img, target = self.data[idx], self.targets[idx]

        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)
        return img, target
    
if __name__ == "__main__":
    # visualize some images
    from matplotlib import pyplot as plt
    dataset = FashionMNISTDataset(root="./data", train=True)
    classes = {
        "T-shirt/top": 0,
        "Trouser": 1,
        "Pullover": 2,
        "Dress": 3,
        "Coat": 4,
        "Sandal": 5,
        "Shirt": 6,
        "Sneaker": 7,
        "Bag": 8,
        "Ankle boot": 9,
    }

    n_images_per_class = 5
    fig, axes = plt.subplots(nrows=len(classes), ncols=n_images_per_class, figsize=(10, 15))
    for i, (class_name, class_idx) in enumerate(classes.items()):
        class_images = [
            dataset[j][0].numpy().reshape(28, 28)
            for j in range(len(dataset))
            if dataset[j][1] == class_idx
        ]
        for j in range(n_images_per_class):
            axes[i, j].imshow(class_images[j], cmap="gray")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].spines["top"].set_visible(False)
            axes[i, j].spines["right"].set_visible(False)
            axes[i, j].spines["bottom"].set_visible(False)
            axes[i, j].spines["left"].set_visible(False)

        axes[i, 0].set_ylabel(class_name, rotation=90, size="large", labelpad=10)
    plt.subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.show()
        