# CVAE Demo

A Conditional Variational Autoencoder (CVAE) implementation for controllable Fashion-MNIST image generation.

## What is CVAE?

CVAE (Conditional Variational Autoencoder) extends the VAE by adding class conditioning, allowing controlled generation of specific types of images.

### Architecture Comparison:

- **Autoencoder (AE)**: `Input → Encoder → Latent → Decoder → Output`
  - Deterministic latent representation
  - Can reconstruct but struggles to generate new samples

- **Variational Autoencoder (VAE)**: `Input → Encoder → μ,σ → Sample z → Decoder → Output`
  - Probabilistic latent space (μ, σ parameters)
  - Can generate new samples by sampling from latent space
  - Uses KL divergence to regularize latent distribution

- **Conditional VAE (CVAE)**: `Input + Class → Encoder → μ,σ → Sample z → Decoder + Class → Output`
  - **Key advantage**: Can control what type of image to generate
  - Example: "Generate a T-shirt" vs "Generate a shoe"
  - Conditioning happens in both encoder and decoder

## Quick Start
### 1. Create the virtual environment with Python 3.13
```bash
conda create -n cvae-demo python=3.13
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Hyperparameter Optimization
```bash
python src/optuna_search.py --n-trials 20 --epochs 10
```

### 3.5. Run Optuna Dashboard to understand the optimization results (Optional)
```bash
optuna-dashboard sqlite:///optuna.db
```

### 4. Train the Model
```bash
python src/train.py --epochs 25
```

### 5. Run the Demo
```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501 to generate the images based on the selected class.

## Project Structure
```
cvae-demo/
├── src/                    # Core model and training code
├── streamlit_app/          # Interactive web interface with Streamlit
├── weights/                # Trained model checkpoints
├── data/                   # Fashion-MNIST dataset
└── src/config/             # Hyperparameter configurations
```

## Architecture Details

### Model Structure
**Fully Connected CVAE** (784→400→20→400→784)

**Encoder**:
- Input: Flattened 28×28 image (784) + one-hot class label (10)
- Hidden: Linear layer with 400 neurons + ReLU
- Output: μ (mean) and log σ² (log variance) vectors (20 dimensions each)

**Reparameterization**: `z = μ + σ × ε` where `ε ~ N(0,1)`

**Decoder**:
- Input: Sampled latent vector z (20) + one-hot class label (10) 
- Hidden: Linear layer with 400 neurons + ReLU
- Output: Reconstructed image (784) + Sigmoid activation

**Loss Function**: `L = BCE(x, x̂) + KL(q(z|x,y) || p(z))`
- BCE: Binary cross-entropy for reconstruction
- KL: Kullback-Leibler divergence for latent regularization

**Fashion-MNIST Classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Comming Soon
- **Evaluates model quality** with classifier accuracy and t-SNE visualization