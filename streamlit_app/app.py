import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import json
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from model import CVAE
from utils import one_hot_encode
from dataset import FashionMNISTDataset
from dataloader import DataLoader

# Configuration
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Load the trained CVAE model"""
    with open('src/config/best_hyperparams.json', 'r') as f:
        config = json.load(f)
    model = CVAE(hidden_dim=config.get("hidden_dim", 400), latent_dim=config.get("latent_dim", 20))

    model_paths = [
        "weights/last_model.pth"
    ]
    
    model_loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                st.success(f"Loaded trained model from {path}")
                model_loaded = True
                break
            except Exception as e:
                st.warning(f"Failed to load {path}: {e}")
                continue
    
    if not model_loaded:
        st.warning("No trained model found")
        raise ValueError("No trained model found")
    
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_data
def load_test_data():
    """Load Fashion-MNIST test data for classifier evaluation"""
    dataset = FashionMNISTDataset(root="./data", train=False)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
    
    all_data = []
    all_labels = []
    
    for data, labels in dataloader:
        all_data.append(data)
        all_labels.append(labels)
    
    return torch.cat(all_data, dim=0), torch.cat(all_labels, dim=0)

def tensor_to_image(tensor):
    """Convert tensor to PIL Image for display"""
    # Handle different tensor shapes from the fully connected model
    if tensor.dim() == 1:
        # Single flattened image (784,) -> reshape to (28, 28)
        if tensor.numel() == 784:
            tensor = tensor.view(28, 28)
        else:
            raise ValueError(f"Expected tensor size 784, got {tensor.numel()}")
    elif tensor.dim() == 2:
        # Batch of flattened images (batch_size, 784) -> take first and reshape
        if tensor.shape[1] == 784:
            tensor = tensor[0].view(28, 28)
        elif tensor.shape == (28, 28):
            # Already correct shape
            pass
        else:
            raise ValueError(f"Unexpected 2D tensor shape: {tensor.shape}")
    elif tensor.dim() == 3:
        # (1, 28, 28) or (28, 28, 1) -> squeeze to (28, 28)
        tensor = tensor.squeeze()
        if tensor.shape != (28, 28):
            raise ValueError(f"After squeezing 3D tensor, got shape {tensor.shape}, expected (28, 28)")
    elif tensor.dim() == 4:
        # (1, 1, 28, 28) -> squeeze to (28, 28)
        tensor = tensor.squeeze()
        if tensor.shape != (28, 28):
            raise ValueError(f"After squeezing 4D tensor, got shape {tensor.shape}, expected (28, 28)")
    else:
        raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")
    
    # Ensure tensor is in [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and scale to [0, 255]
    np_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    # Create PIL image
    image = Image.fromarray(np_array, mode='L')
    return image

def generate_samples(model, class_label, num_samples=9):
    """Generate samples for a specific class"""
    with torch.no_grad():
        # Create random latent vectors
        z = torch.randn(num_samples, model.latent_dim).to(DEVICE)
        
        # Create condition (one-hot encoded class)
        labels = torch.full((num_samples,), class_label, dtype=torch.long)
        condition = one_hot_encode(labels, num_classes=10).to(DEVICE)
        
        # Generate samples using the decoder
        samples = model.decode(z, condition)
        
        # Samples are now (num_samples, 784) - flattened images
        # Reshape each sample to (28, 28) for display
        samples_reshaped = samples.view(num_samples, 28, 28)
        
        return samples_reshaped

def create_image_grid(images, cols=3):
    """Create a grid of images for display"""
    rows = len(images) // cols + (1 if len(images) % cols > 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="CVAE Interactive Demo",
        layout="wide"
    )
    
    st.title("CVAE Interactive Demo")
    
    # Load model
    st.sidebar.header("Model Status")
    model = load_model()
    st.sidebar.write(f"Device: {DEVICE}")
    st.sidebar.write(f"Latent Dimension: {model.latent_dim}")
    
    # Main tabs
    tab1, tab2 = st.tabs(["Interactive Generation", "Clusters Visualization"])
    
    with tab1:
        st.header("Interactive Generation")
        st.write("Generate Fashion-MNIST samples by selecting a class and adjusting parameters.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Class selection
            selected_class = st.selectbox(
                "Select Class:",
                options=range(len(CLASS_NAMES)),
                format_func=lambda x: f"{x}: {CLASS_NAMES[x]}",
                index=0
            )
            
            # Number of samples
            num_samples = st.slider(
                "Number of Samples:",
                min_value=1,
                max_value=16,
                value=9,
                step=1
            )
            
            # Grid columns
            grid_cols = st.slider(
                "Grid Columns:",
                min_value=1,
                max_value=4,
                value=3,
                step=1
            )
            
            # Generate button
            if st.button("Generate Samples", type="primary"):
                st.session_state.generate_new = True
            
            # Auto-generate on first load
            if 'generate_new' not in st.session_state:
                st.session_state.generate_new = True
        
        with col2:
            if st.session_state.get('generate_new', False):
                with st.spinner("Generating samples..."):
                    # Generate samples
                    samples = generate_samples(model, selected_class, num_samples)
                    
                    # Convert to images
                    images = [tensor_to_image(sample) for sample in samples]
                    
                    # Create and display grid
                    fig = create_image_grid(images, cols=grid_cols)
                    st.pyplot(fig)
                    
                    st.success(f"Generated {num_samples} samples for class: {CLASS_NAMES[selected_class]}")
                
                st.session_state.generate_new = False
    
    with tab2:
        st.header("Latent Space Visualization")
        st.write("t-SNE visualization of latent space colored by class")
        st.write("Coming soon...")
        
    

if __name__ == "__main__":
    main()