import os 
from PIL import Image
import timm
import json 
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch 
import torch.nn as nn
import torch.nn.functional as F


layer_translate = {'resnet50': 'fc',
                   'efficientnet_b0': 'classifier'}

dict_mapping = {'resnet50': "isic_resnet50",
                'efficientnet_b0': "effnetb0_isic"}

class NeuralNetwork(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.encoder = timm.create_model(model_name, pretrained=True)
        self.clf_layer = layer_translate[model_name]
        self.classifier = self.set_classifier()
        self.load_model()
        self.encoder.eval()
        self.classifier.eval()
    
    def load_model(self):
        ckpt = torch.load(os.path.join(os.getcwd(), 'ckpts', f"{dict_mapping[self.model_name]}.ckpt"), map_location='cpu')
        msg = self.load_state_dict(ckpt)
        print(f"Loading msg: {msg=}")

    def set_classifier(self):
        in_features = getattr(self.encoder, self.clf_layer).in_features
        #override the current classification head
        setattr(self.encoder, self.clf_layer, nn.Identity())
        return nn.Linear(in_features, 2)

    def forward(self, x):
        features = self.encoder(x)
        # print(f"{features.shape=}")
        logits = self.classifier(features)
        # print(f"{logits.shape=}")
        return logits
        

class ImageClassifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = NeuralNetwork(model_name)
        self.transforms = create_transform(**resolve_data_config(self.model.encoder.pretrained_cfg, 
                                                                 model=self.model.encoder))
        # Load pretrained_checkpoint 
        self.id2label = {
                        0: "Benigno",
                        1: "Melanoma",
                         }
    
    def load_ckpt():
        dict_mapping = {"": "isic_resnet50",
                        "": "effnetb0_isic"}

    def predict(self, image):
        inputs = self.transforms(image).unsqueeze(0)
        # Create extra dimensions for batching 

        print(inputs.shape)

        with torch.no_grad():
            logits = self.model(inputs)

        probs = F.softmax(logits, dim=-1)[0] # remove batch 
        # print(f"{probs.shape=}/ {probs=}")

        scores = []
        for i, prob in enumerate(probs):
            scores.append({
                "score": prob.item(),
                "label": self.id2label[i]
            })
        # print(scores)
        return scores



if __name__ == "__main__":
    from huggingface_hub import login
    login()
    resnet_name = 'resnet50'
    ef_name = 'efficientnet_b0'

    curr_model = resnet_name
    model = ImageClassifier(curr_model)

    dict_mapping = {resnet_name: "isic_resnet50",
                    ef_name: "effnetb0_isic"}

    # ckpt = torch.load(os.path.join(os.getcwd(), 'ckpts', f"{dict_mapping[curr_model]}.ckpt"), map_location='cpu')
    # ckpt = torch.load(os.path.join(os.getcwd(), 'ckpts', f"{dict_mapping[curr_model]}.ckpt"), map_location='cpu')
    # msg = model.model.load_state_dict(torch.load('./isic_resnet50.ckpt'))
    
    # torch.save(ckpt['state_dict'], './isic_resnet50.ckpt')
    # print(model.transforms)

    # sample_img_path = '/Users/levygurgel/workspace/Datasets/isic2019/ISIC_2019_Training_Input/ISIC_0064481.jpg'  # -> duvida
    sample_img_path = '/Users/levygurgel/workspace/Datasets/isic2019/ISIC_2019_Training_Input/ISIC_0054728.jpg' # -> 0.98 melanoma
    # sample_img_path = '/Users/levygurgel/workspace/Datasets/isic2019/ISIC_2019_Training_Input/ISIC_0027896.jpg' # 
    img = Image.open(sample_img_path).convert('RGB')

    predictions = model.predict(img)
