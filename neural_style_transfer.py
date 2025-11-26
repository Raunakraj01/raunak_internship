import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- IMAGE LOADER -----------------------
def load_image(path, max_size=512):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((max_size, max_size)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)


def pil_from_tensor(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


# ----------------------- GET FEATURES -----------------------
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)


# ----------------------- GUI FUNCTIONS -----------------------
content_path = ""
style_path = ""

def pick_content():
    global content_path
    content_path = filedialog.askopenfilename(title="Select Content Image",
                                              filetypes=[("Images", "*.jpg *.png")])
    if content_path:
        content_label.config(text=f"Content: {content_path.split('/')[-1]}")


def pick_style():
    global style_path
    style_path = filedialog.askopenfilename(title="Select Style Image",
                                            filetypes=[("Images", "*.jpg *.png")])
    if style_path:
        style_label.config(text=f"Style: {style_path.split('/')[-1]}")


def start_transfer():
    global content_path, style_path

    if content_path == "" or style_path == "":
        messagebox.showwarning("Missing File", "Please select both content and style images.")
        return

    try:
        status_box.delete("1.0", tk.END)
        status_box.insert(tk.END, "Loading images...\n")

        content = load_image(content_path)
        style = load_image(style_path)

        vgg = models.vgg19(pretrained=True).features.to(device).eval()

        for param in vgg.parameters():
            param.requires_grad_(False)

        layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # content
            "28": "conv5_1"
        }

        content_features = get_features(content, vgg, layers)
        style_features = get_features(style, vgg, layers)

        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        target = content.clone().requires_grad_(True).to(device)

        optimizer = optim.Adam([target], lr=0.003)

        style_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.75,
            'conv3_1': 0.2,
            'conv4_1': 0.2,
            'conv5_1': 0.2
        }
        content_weight = 1e4
        style_weight = 1e2

        status_box.insert(tk.END, "Starting style transfer...\n")

        for step in range(301):
            target_features = get_features(target, vgg, layers)

            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

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

            if step % 50 == 0:
                status_box.insert(tk.END, f"Step {step} | Loss: {total_loss.item():.2f}\n")
                status_box.see(tk.END)
                root.update_idletasks()

        final_img = pil_from_tensor(target)

        final_img.save("styled_output.jpg")
        status_box.insert(tk.END, "\nDone! Saved as styled_output.jpg")

        # Show final image in GUI
        final_img_tk = ImageTk.PhotoImage(final_img.resize((300, 300)))
        img_canvas.config(image=final_img_tk)
        img_canvas.image = final_img_tk

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ----------------------- GUI WINDOW -----------------------
root = tk.Tk()
root.title("Neural Style Transfer GUI")
root.geometry("850x600")
root.resizable(False, False)

title = tk.Label(root, text="Neural Style Transfer", font=("Arial", 18, "bold"))
title.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

content_btn = tk.Button(btn_frame, text="Select Content Image", font=("Arial", 12), command=pick_content)
content_btn.grid(row=0, column=0, padx=10)

style_btn = tk.Button(btn_frame, text="Select Style Image", font=("Arial", 12), command=pick_style)
style_btn.grid(row=0, column=1, padx=10)

content_label = tk.Label(btn_frame, text="Content: None", font=("Arial", 10))
content_label.grid(row=1, column=0)

style_label = tk.Label(btn_frame, text="Style: None", font=("Arial", 10))
style_label.grid(row=1, column=1)

start_btn = tk.Button(root, text="Start Style Transfer", font=("Arial", 14), command=start_transfer)
start_btn.pack(pady=10)

status_label = tk.Label(root, text="Progress:", font=("Arial", 12))
status_label.pack()

status_box = tk.Text(root, height=10, width=100, font=("Arial", 10))
status_box.pack()

img_canvas = tk.Label(root)
img_canvas.pack(pady=10)

root.mainloop()
