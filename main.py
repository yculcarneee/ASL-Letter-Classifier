from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import cv2 
import time

@st.cache(allow_output_mutation=True)
def ListObj():
    return []

def get_predictions(img, model):

    probs = nn.Softmax(dim=1)
    
    img = img.unsqueeze(0)
    
    top5_classes = []
    
    with torch.no_grad():
        model.eval()

        output = model(img)
        softmax_output = probs(output)   
        
        top5_probs, top5_label = torch.topk(softmax_output, 5)
        
        top5_probs_list = top5_probs.cpu().flatten().tolist()
        top5_label_list = top5_label.cpu().flatten().tolist()
        
        for elem in top5_label_list:
            top5_classes.append(index_to_char_map[elem])
            
        return top5_probs_list, top5_classes

@st.cache
def load_model():
    device = torch.device('cpu')
    model = models.vgg16_bn()
    model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5, inplace=False),
                                     nn.Linear(in_features=4096, out_features=4096, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=0.5, inplace=False),
                                     nn.Linear(in_features=4096, out_features=29, bias=True)
                                    )
    model.load_state_dict(torch.load(save_file_name, map_location=device))

    return model

if __name__ == "__main__":

    index_to_char_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

    save_file_name = #ENTER MODEL PATH HERE

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error")

    st.title("ASL Letter Classifier")
    st.subheader("Press the Capture button to classify current image")

    frameST = st.empty()

    img_counter = 0

    image_transforms = {
        'test':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    model = load_model()

    special_letters = ["del", "space", "nothing"]

    listObj = ListObj()

    flag = st.button('Capture')
    
    while cam.isOpened(): 
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        cropped_frame = frame[102:298, 402:598]
        
        cv2.rectangle(frame, (400, 100), (600, 300), (255, 0, 0), 2)
        cv2.imwrite("temp.png", frame)
        temp_img = Image.open("temp.png")
        if not ret:
            break
        time.sleep(0.05) #ADJUST THIS VALUE AS PER YOUR DISK DRIVE SPEED
        frameST.image(temp_img, channels="BGR")

        if(flag):
            flag = False

            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, cropped_frame)
            img = Image.open(img_name)
            transformed_img = image_transforms['test'](img)
            top5_prob, top5_label = get_predictions(transformed_img, model)

            if top5_label[0] not in special_letters:
                listObj.append(top5_label[0])

            if top5_label[0] == "del":
                listObj.pop()

            if top5_label[0] == "space":
                listObj.append(" ")
            
            text = ""

            for elem in listObj:
                text += elem
            
            st.write(text)

            plt.figure(figsize=(16, 10))

            plt.bar(top5_label, top5_prob)
            plt.title('Top 5 Probabilities', fontsize=20)
            plt.xlabel('Classes', fontsize=20)
            plt.ylabel('Probabilities', fontsize=20)
    
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            st.pyplot()
            
            img_counter += 1  
            
    cam.release()
