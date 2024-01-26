import numpy as np
from PIL import Image
import torch


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

# For RGB image inputs
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

def load_entire_model(filepath):
    checkpoint = torch.jit.load(filepath)
    metadata = checkpoint.metadata

    return checkpoint, metadata

if __name__=="__main__":
    # Inference for a single image
    #### Method 2: Torchscript Saving & Loading #####

    image_path = "Final_results/test_images/mittel_0104_Kombination_20_num_2_1_t40.jpg"
    inp_tensor = preprocessing_v2(image_path)
    inp_tensor = inp_tensor.reshape(1, *inp_tensor.shape) # reshaping N, C, H, W. N is batch size.

    labels = ["Error", "Non Error"]
    filepath = "Final_results/Cable_epoch=29-step=5610_grayscale_v2_torchscript_model_with_metadata.ckpt"

    model, metadata = load_entire_model(filepath)
    pred = model(inp_tensor)
    labels = ["Error", "Non Error"]
    print(pred)
    pred = torch.argmax(pred, dim=1)
    print(f"Label of image is {labels[pred[0]]}")