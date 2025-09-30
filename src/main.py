import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from tis import TIS
from metrics.insertion import Insertion
from metrics.deletion import Deletion
from metrics.sparseness import Sparseness
from metrics.sensitivity import SensitivityMax
from skimage.transform import resize
import random
import warnings
from tqdm import tqdm
import logging
import torch.nn as nn
import csv
import pandas as pd
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")
dataset_root = 'data'
noise_types = ['brightness', 'gaussian_blur', 'gaussian_noise', 'motion_blur', 'shot_noise']
severity_level = 5
conditions = [('clean', None)] + [(noise, severity_level) for noise in noise_types]
xai_methods = ['rise', 'tis']  # Only RISE and TIS
output_dir = 'results'
saliency_dir = f'{output_dir}/saliency_maps'
metrics_dir = f'{output_dir}/metrics'
predictions_file = f'{output_dir}/predictions.csv'  # File for predictions
subset_size = 10  # Use exactly 10 images
input_size = (224, 224)  # Input size for RISE
n_masks_rise = 1000  # Optimized number of RISE masks
n_masks_tis = 256  # Optimized number of TIS masks
gpu_batch = 64  # Optimized batch size for RISE

# Create output directories
logger.info("Creating output directories")
os.makedirs(saliency_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
for metric in ['insertion', 'deletion', 'sparseness', 'sensitivity']:
    os.makedirs(f'{metrics_dir}/{metric}', exist_ok=True)

# Initialize predictions CSV file if it doesn't exist
if not os.path.exists(predictions_file):
    logger.info(f"Initializing predictions file at {predictions_file}")
    with open(predictions_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'condition', 'model', 'predicted_class'])

# Dataset and DataLoader
class ImageNetCDataset:
    def __init__(self, root_dir, transform=None, subset_size=None, selected_images=None, condition=None):
        self.condition = condition
        all_paths = [os.path.join(root, f) for root, _, files in os.walk(root_dir) for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
        if selected_images:
            if condition[0] == 'clean' or condition[1] is None:
                # For clean images, match exact filenames
                self.image_paths = [p for p in all_paths if os.path.basename(p) in selected_images]
            else:
                # For corrupted images, match core filename (e.g., 101_ILSVRC2012_val_00000067.JPEG)
                self.image_paths = []
                for p in all_paths:
                    filename = os.path.basename(p)
                    # Match core filename, preserving prefix, stripping corruption suffix
                    core_match = re.match(r'(.*?)(?:_(brightness|gaussian_blur|gaussian_noise|motion_blur|shot_noise)_sev\d)\.(jpg|jpeg|png|JPEG)', filename)
                    if core_match:
                        core_filename = core_match.group(1) + '.' + core_match.group(3)
                        if core_filename in selected_images:
                            self.image_paths.append(p)
        else:
            self.image_paths = sorted(all_paths)[:subset_size]
        logger.info(f"Found {len(self.image_paths)} images in {root_dir}")
        if len(self.image_paths) < subset_size:
            logger.warning(f"Only {len(self.image_paths)} images found, expected {subset_size}")
            logger.info(f"Available files in {root_dir}: {[os.path.basename(p) for p in all_paths]}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[idx]

# Select 10 specific clean images (first 10 by default)
clean_dir = f'{dataset_root}/clean'
all_clean_images = [os.path.basename(f) for f in os.listdir(clean_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
selected_images = sorted(all_clean_images)[:subset_size]  # First 10 images
logger.info(f"Selected images: {selected_images}")

transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# RISE implementation
class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        logger.info(f"Generating {N} RISE masks")
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating masks'):
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        torch.save(torch.from_numpy(self.masks).float(), savepath)  # Save as .pt
        self.masks = torch.from_numpy(self.masks).float().to(device)
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        logger.info(f"Loading RISE masks from {filepath}")
        self.masks = torch.load(filepath).to(device)
        self.N = self.masks.shape[0]
        self.p1 = 0.5  # Set default p1, matching generate_masks

    def forward(self, x):
        logger.info(f"Applying RISE to image with shape: {x.shape}")
        N = self.N
        _, C, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)  # (N, 1, H, W) * (1, C, H, W) -> (N, C, H, W)
        logger.info(f"Stack shape: {stack.shape}")

        # Process in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            batch_stack = stack[i:min(i + self.gpu_batch, N)].to(device)
            logger.info(f"Batch stack shape: {batch_stack.shape}")
            with torch.no_grad():
                batch_preds = torch.softmax(self.model(batch_stack), dim=1)
            p.append(batch_preds)  # Keep on GPU
        p = torch.cat(p)  # (N, num_classes)
        
        # Compute saliency map
        CL = p.size(1)
        sal = torch.matmul(p.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view(CL, H, W)
        sal = sal / N / self.p1
        logger.info(f"Saliency shape: {sal.shape}")
        return sal

# Visualization
def plot_saliency_map(image, saliency_map, title, save_path):
    logger.info(f"Saving saliency map to {save_path}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    # Denormalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image.squeeze(0).permute(1, 2, 0).cpu().numpy() * std + mean
    img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Normalize saliency map and apply colormap
    saliency = saliency_map.cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)  # Normalize to 0–1
    heatmap = np.uint8(cm.jet(saliency)[..., :3] * 255)  # Apply jet colormap
    plt.imshow(img)  # Show original image
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
    plt.title(title)
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# Metric computation
def compute_metrics(image, saliency_map, model, target, model_name, method, condition, image_idx):
    logger.info(f"Computing metrics for {model_name} - {method} - image {image_idx}")
    results = {}
    insertion_metric = Insertion(model, n_steps=100, batch_size=32, baseline='blur')
    deletion_metric = Deletion(model, n_steps=100, batch_size=32, baseline='blur')
    sparseness_metric = Sparseness(shift=True)
    sensitivity_metric = SensitivityMax(model, method=lambda x, class_idx: xai_instances[method][model_name](x)[class_idx] if method == 'rise' else xai_instances[method][model_name](x, class_idx=class_idx), n_perturb_samples=10, perturb_radius=0.02)

    # Insertion and Deletion
    logger.info("Computing insertion metric")
    insertion_scores = insertion_metric(image, saliency_map, target=target)
    logger.info("Computing deletion metric")
    deletion_scores = deletion_metric(image, saliency_map, target=target)
    results['insertion'] = insertion_scores.cpu().numpy()
    results['deletion'] = deletion_scores.cpu().numpy()

    # Sparseness
    logger.info("Computing sparseness metric")
    saliency_map = saliency_map.to(device)
    sparseness_score = sparseness_metric(image.to(device), saliency_map)
    results['sparseness'] = sparseness_score.detach().cpu().numpy()

    # Sensitivity
    logger.info("Computing sensitivity metric")
    sensitivity_score = sensitivity_metric(image.to(device), target=target)
    results['sensitivity'] = sensitivity_score.detach().cpu().numpy()

    # Save metric results
    condition_name = condition[0] if condition[1] is None else f'{condition[0]}_{condition[1]}'
    for metric_name, scores in results.items():
        save_path = f'{metrics_dir}/{metric_name}/{condition_name}_{image_idx}_{model_name}_{method}.npy'
        logger.info(f"Saving {metric_name} scores to {save_path}")
        np.save(save_path, scores)

    # Plot insertion/deletion curves
    logger.info(f"Saving insertion/deletion plot for {condition_name}_{image_idx}_{model_name}_{method}")
    plt.figure(figsize=(10, 5))
    plt.plot(np.linspace(0, 1, len(results['insertion'])), results['insertion'], label='Insertion')
    plt.plot(np.linspace(0, 1, len(results['deletion'])), results['deletion'], label='Deletion')
    plt.title(f'{model_name} - {method} - {condition_name}')
    plt.xlabel('Fraction of Pixels')
    plt.ylabel('Model Confidence')
    plt.legend()
    plt.savefig(f'{metrics_dir}/insertion_deletion_{condition_name}_{image_idx}_{model_name}_{method}.png')
    plt.close()

# Check if all outputs exist for an image
def outputs_exist(image_path, condition_name, model_name, image_idx, methods=['rise', 'tis']):
    if os.path.exists(predictions_file):
        df = pd.read_csv(predictions_file)
        if not df[(df['image_path'] == image_path) & (df['condition'] == condition_name) & (df['model'] == model_name)].empty:
            for method in methods:
                saliency_path = f'{saliency_dir}/{condition_name}_{image_idx}_{model_name}_{method}.png'
                if not os.path.exists(saliency_path):
                    return False
                for metric in ['insertion', 'deletion', 'sparseness', 'sensitivity']:
                    metric_path = f'{metrics_dir}/{metric}/{condition_name}_{image_idx}_{model_name}_{method}.npy'
                    if not os.path.exists(metric_path):
                        return False
                plot_path = f'{metrics_dir}/insertion_deletion_{condition_name}_{image_idx}_{model_name}_{method}.png'
                if not os.path.exists(plot_path):
                    return False
            return True
    return False

# Model configurations
model_configs = [
    ('vit_b16', lambda: create_model(model_name='vit_base_patch16_224', pretrained=True, pretrained_cfg='orig_in21k_ft_in1k').eval().to(device))
]

# Experiment Loop
for model_name, model_fn in model_configs:
    logger.info(f"Starting processing for model: {model_name}")
    model = model_fn()  # Initialize model
    xai_instances = {
        'rise': {model_name: RISE(model, input_size=input_size, gpu_batch=gpu_batch)},
        'tis': {model_name: TIS(model, n_masks=n_masks_tis)}
    }
    # Handle RISE mask generation
    mask_path = f'masks_{model_name}.pt'
    if not os.path.exists(mask_path):
        logger.info(f'Generating RISE masks for {model_name} at {mask_path}')
        xai_instances['rise'][model_name].generate_masks(N=n_masks_rise, s=7, p1=0.5, savepath=mask_path)
    else:
        xai_instances['rise'][model_name].load_masks(mask_path)

    for condition in conditions:
        noise_type, severity = condition
        root_dir = f'{dataset_root}/clean' if noise_type == 'clean' else f'{dataset_root}/imagenet_c/{noise_type}/{severity}'
        logger.info(f"Loading dataset for condition: {noise_type}{'' if severity is None else f'_{severity}'}")
        dataset = ImageNetCDataset(root_dir, transform=transform, subset_size=subset_size, selected_images=selected_images, condition=condition)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        condition_name = noise_type if severity is None else f'{noise_type}_{severity}'
        logger.info(f'Processing condition: {condition_name}')
        
        for i, (image, image_path) in enumerate(tqdm(dataloader, total=len(dataset), desc=f'{model_name} - {condition_name}')):
            if outputs_exist(image_path[0], condition_name, model_name, i):
                logger.info(f"Outputs already exist for {image_path[0]} in condition {condition_name}, skipping")
                continue
            logger.info(f"Processing image {i+1}/{len(dataset)}: {image_path}")
            image = image.to(device)
            pred_class = torch.argmax(model(image)).item()
            logger.info(f"Predicted class for {image_path}: {pred_class}")
            
            # Save prediction to CSV
            with open(predictions_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([image_path[0], condition_name, model_name, pred_class])

            for method in xai_methods:
                logger.info(f"Generating saliency map for method: {method}")
                # Generate saliency map
                if method == 'rise':
                    saliency = xai_instances['rise'][model_name](image)[pred_class]
                elif method == 'tis':
                    saliency = xai_instances['tis'][model_name](image, class_idx=pred_class).to(device)
                
                # Ensure saliency map is 224x224
                if saliency.shape != (224, 224):
                    logger.info(f"Resizing saliency map for {method} to 224x224")
                    saliency = torch.nn.functional.interpolate(saliency.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze().to(device)
                
                # Save saliency map
                save_path = f'{saliency_dir}/{condition_name}_{i}_{model_name}_{method}.png'
                plot_saliency_map(image, saliency, f'{model_name} - {method} - {condition_name}', save_path)

                # Compute and save metrics
                compute_metrics(image, saliency, model, pred_class, model_name, method, condition, i)
    
    # Clear model and XAI instances from memory
    logger.info(f"Clearing model and XAI instances for {model_name}")
    del model
    del xai_instances
    torch.cuda.empty_cache()
    logger.info(f"Finished processing model: {model_name}")
