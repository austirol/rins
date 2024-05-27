import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork

def transform_image(image_in, resize_shape=[256, 256], pre=False):
    if pre:
        image_in = cv2.resize(image_in, (150, 150, 3), cv2.INTER_LINEAR) #cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
    if resize_shape != None:
        image = cv2.resize(image_in, dsize=(resize_shape[1], resize_shape[0]))

    image = image / 255.0
    
    image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)

    image = np.transpose(image, (2, 0, 1))
    
    return image

# real_image_path = os.path.join(os.getcwd(), "data", "real", "test", "fake", "real_0.png")
# real_image = cv2.imread(os.path.join(os.getcwd(), "data", "real", "test", "fake", "real_0.png"), cv2.IMREAD_COLOR)

# checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
# run_name = "DRAEM_test_0.0001_720_bs8_real_"

# model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
# model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
# model.cuda()
# model.eval()

# model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
# model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
# model_seg.cuda()
# model_seg.eval()

# image = transform_image(real_image)
# tensor_image = torch.from_numpy(np.array([image]))

# image_cuda = tensor_image.cuda()
# # print(image_cuda.shape)

# result_class = model(image_cuda)
# joined = torch.cat((result_class.detach(), image_cuda), dim=1)
# out_mask = model_seg(joined)
# result_mask = torch.softmax(out_mask, dim=1)
# result_mask_cv = result_mask[0 ,1 ,: ,:].detach().cpu().numpy()

# out_mask_averaged = torch.nn.functional.avg_pool2d(result_mask[: ,1: ,: ,:], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
# image_score = np.max(out_mask_averaged)

# print(image_score, 0 if image_score < 0.02793829 else 1) #0.02793829

# real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
# result_mask_cv = cv2.resize(result_mask_cv, dsize=(real_image.shape[1], real_image.shape[0]))
# fig = plt.figure(figsize=(15, 5))
# ax = fig.add_subplot(131)
# ax.imshow(real_image)
# ax.imshow(result_mask_cv, alpha = 0.2, vmin=0, vmax=1)
# ax.set_title("Combined")
# ax = fig.add_subplot(132)
# ax.imshow(result_mask_cv, vmin=0, vmax=1)
# ax.set_title("Predicted Mask")
# ax = fig.add_subplot(133)
# ax.imshow(real_image)
# ax.set_title("Original Image")
# # plt.show()


def load_models():
    """Loads the trained models"""
    ### TO DO SPREMENI VSAK
    checkpoint_path = os.path.join("/home/maja/ros/src/dis_tutorial3/scripts", "checkpoints")
    run_name = "DRAEM_test_0.0001_720_bs8_real_"

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
    model.cuda()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()
    
    return model, model_seg


def inference(image, model, model_seg):
    """Classifies the image as real[0] or fake[1]."""

    image = transform_image(image)
    tensor_image = torch.from_numpy(np.array([image]))

    image_cuda = tensor_image.cuda()

    result_class = model(image_cuda)
    joined = torch.cat((result_class.detach(), image_cuda), dim=1)
    out_mask = model_seg(joined)
    result_mask = torch.softmax(out_mask, dim=1)
    result_mask_cv = result_mask[0 ,1 ,: ,:].detach().cpu().numpy()# comment for efficiency

    out_mask_averaged = torch.nn.functional.avg_pool2d(result_mask[: ,1: ,: ,:], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
    image_score = np.max(out_mask_averaged)
    
    prediction = 0 if image_score < 0.016 else 1 # 0.02793829
    
    return prediction, result_mask_cv, image_score

if __name__ == "__main__":
    
    real_image = cv2.imread(os.path.join(os.getcwd(), "data", "real", "test", "good", "real_image4085.png"), cv2.IMREAD_COLOR)

    import time
    
    start = time.time()
    model, model_seg = load_models()
    print("Loading time:", time.time() - start)

    start = time.time()
    label, mask = inference(real_image, model, model_seg)
    print("Inference time:", time.time() - start)
    print("label:", label)

    import matplotlib.pyplot as plt
    plt.imshow(mask)
    plt.show()

    # def test_inference(path, im_list, model, model_seg, truth=None):
    #     predictions = []
    #     for i, im in enumerate(im_list):
    #         # print(i)
    #         image = cv2.imread(os.path.join(path, im), cv2.IMREAD_COLOR)
    #         predictions.append(inference(image, model, model_seg))

    #     predictions = np.array(predictions)
    #     # print(predictions)
    #     # right = len(predictions[predictions == truth]) / len(predictions)
    #     right = len(predictions) - len(predictions[predictions == truth])
    #     return right

    # real_dataset = os.path.join(os.getcwd(), "data", "real", "test", "good")
    # real_list = os.listdir(real_dataset)
    # print(test_inference(real_dataset, real_list, model, model_seg, 0))
    # fake_dataset = os.path.join(os.getcwd(), "data", "real", "test", "fake")
    # fake_list = os.listdir(fake_dataset)
    # print(test_inference(fake_dataset, fake_list, model, model_seg, 1))
    # og_dataset = os.path.join(os.getcwd(), "data", "real", "test", "og")
    # og_list = os.listdir(og_dataset)
    # print(test_inference(og_dataset, og_list, model, model_seg, 1))
    # people_dataset = os.path.join(os.getcwd(), "data", "real", "test", "people")
    # people_list = os.listdir(people_dataset)
    # print(test_inference(people_dataset, people_list, model, model_seg, 1))