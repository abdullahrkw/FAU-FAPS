import numpy as np
from PIL import Image
import torch
from torchvision.models import densenet121

from network import DeepCNN


# This v2 preprocessing is for Cover, Cable and Sheet metal package
# It uses grayscale image rather than RGB image
def preprocessing_v2(image_path, pil_image_mode="L"):
    img = Image.open(image_path)
    img = img.convert(mode=pil_image_mode)
    new_w, new_h = (256,256)
    # resize but keep aspect ratio
    background = Image.new(pil_image_mode, (new_w, new_h))
    img.thumbnail((new_w, new_h))
    background.paste(img, (0,0))
    img = background
    img = np.asarray(img, dtype=np.float32)
    img = (img - np.mean(img))/np.std(img) # normalize
    img = np.expand_dims(img, 0).repeat(3, axis=0)
    img = np.reshape(img, (3, new_h, new_w))
    img = torch.from_numpy(img)
    return img

# This is for RGB inputs
def preprocessing(image_path, pil_image_mode="RGB"):
    img = Image.open(image_path)
    img = img.convert(mode=pil_image_mode)
    new_w, new_h = (256,256)
    # resize but keep aspect ratio
    background = Image.new(pil_image_mode, (new_w, new_h))
    img.thumbnail((new_w, new_h))
    background.paste(img, (0,0))
    img = background
    img = np.asarray(img, dtype=np.float32)
    img = normalize(img)
    img = np.reshape(img, (3, new_h, new_w))
    img = torch.from_numpy(img)
    return img

def normalize(item):
    item = item/255
    # imagenet mean and std
    mean =  np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    item[:,:,0] = (item[:,:,0] - mean[0])/std[0]
    item[:,:,1] = (item[:,:,1] - mean[1])/std[1]
    item[:,:,2] = (item[:,:,2] - mean[2])/std[2]
    return item

def save_entire_model(model, filepath, metadata=None):
    model.metadata = metadata
    model = model.to_torchscript()
    torch.jit.save(model, filepath)

def load_entire_model(filepath):
    checkpoint = torch.jit.load(filepath)
    metadata = checkpoint.metadata

    return checkpoint, metadata


num_classes = 2
model = densenet121()

# fc for resnet, classifier for densenet
backbone_out_features = model.classifier.out_features

final_model = DeepCNN(backbone=model,
                        backbone_out=backbone_out_features,
                        num_classes=num_classes,
                        views=["file_name"])

device="cpu"
model_path = "Final_results/Sheet_Metal_Package_epoch=29-step=3750_grayscale_v2.ckpt"
checkpoint = torch.load(model_path, map_location=torch.device(device))
final_model.load_state_dict(checkpoint["state_dict"])
final_model.eval()

# Inference for a single image
image_path = "Final_results/test_images/mittel_0104_Kombination_20_num_2_1_t40.jpg"
inp_tensor = preprocessing_v2(image_path)
inp_tensor = inp_tensor.reshape(1, *inp_tensor.shape) # reshaping N, C, H, W. N is batch size.
pred = final_model(inp_tensor)

labels = ["Error", "Non Error"]
print(pred)
pred = torch.argmax(pred, dim=1)
print(f"Label of image is {labels[pred[0]]}")

#### Method 2: Torchscript Saving & Loading #####

metadata = {'input_size': 256, 'num_classes': num_classes}
filepath = "Final_results/Sheet_Metal_Package_epoch=29-step=3750_grayscale_v2_torchscript_model_with_metadata.ckpt"
save_entire_model(final_model, filepath, metadata=metadata)

model, metadata = load_entire_model(filepath)
pred = model(inp_tensor)
labels = ["Error", "Non Error"]
print(pred)
pred = torch.argmax(pred, dim=1)
print(f"Label of image is {labels[pred[0]]}")