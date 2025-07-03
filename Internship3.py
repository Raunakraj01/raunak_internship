"""
Neural Style Transfer Script
This blends a content image with an artistic style image using VGG19.
VGG19 is used because it captures image features well.
Gram matrix helps represent texture/style by computing relationships between feature map
if u are testing this please make sure to download libraries in appropriate file path
The result improves after each iteration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and resize image
def load_image(path, max_size=512):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Display image
def imshow(tensor, title=""):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Get features from specific layers
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix for style representation
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# Load images
content = load_image("content.jpg")
style = load_image("style.jpg")

# Load VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze model parameters
for param in vgg.parameters():
    param.requires_grad_(False)

# Layers we care about
layers = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "21": "conv4_2",  # content layer
    "28": "conv5_1"
}

# Extract features
content_features = get_features(content, vgg, layers)
style_features = get_features(style, vgg, layers)

# Style Gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image
target = content.clone().requires_grad_(True).to(device)

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Loss weights
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}
content_weight = 1e4
style_weight = 1e2

# Training loop
print("Starting style transfer...")
for step in range(301):
    target_features = get_features(target, vgg, layers)

    # Compute content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # Compute style loss
    style_loss = 0
    for layer in style_weights:
        target_feat = target_features[layer]
        target_gram = gram_matrix(target_feat)
        style_gram = style_grams[layer]
        style_loss += style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Show progress
    if step % 50 == 0:
        print(f"Step {step}, Loss: {total_loss.item():.2f}")
        imshow(target, title=f"Step {step}")

# Save final result
final_img = target.clone().squeeze()
final_img = transforms.ToPILImage()(final_img.cpu())
final_img.save("styled_output.jpg")
print("✅ Style transfer completed and saved as styled_output.jpg!")