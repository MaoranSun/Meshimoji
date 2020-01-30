import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

####Face detection

#Load cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Emotion types
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
#Init Camera
cap = cv2.VideoCapture(0)


def crop_face(frame):
    #Input: img
    #Output: the face with largest areas in the img

    #Change to grayscale img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Use cascade classifier to capture all faces in the img
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Check if face exists in the img
    if len(faces) == 0:
        return np.array([None, None])

    #Find the largest face
    max_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_face[2] * max_face[3]:
            max_face = face

    #Crop the largest face 
    crop_img = gray[max_face[1] : max_face[1] + max_face[3], max_face[0] : max_face[0] + max_face[2]]
    
    #Resize the face for CNN classifier
    resized_img = cv2.resize(crop_img, (48, 48))


    return resized_img


###Send image to model for classification

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(1, 64, 3, padding = 1)
        self.conv1 = nn.Conv2d(64, 64, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout()
        self.batch1 = nn.BatchNorm2d(64)
        self.batch2 = nn.BatchNorm2d(128)
        self.batch3 = nn.BatchNorm2d(256)
        self.batch4 = nn.BatchNorm2d(512)
        self.flat = nn.Flatten()
        
        self.conv02 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv03 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv04 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding = 1)
        
        self.fc1 = nn.Linear(in_features=4608, out_features=256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 7)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = F.relu(self.conv02(x))
        x = self.batch2(x)
        x = F.relu(self.conv2(x))
        x = self.batch2(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = F.relu(self.conv03(x))
        x = self.batch3(x)
        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = F.relu(self.conv04(x))
        x = self.batch4(x)
        x = F.relu(self.conv4(x))
        x = self.batch4(x)
        x = self.pool(x)
        x = self.drop(x)
        
        x = self.flat(x)
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        
        x = F.softmax(self.fc4(x), dim=1)
        
        return x

#Load CNN Classifier
net = Net()
net.load_state_dict(torch.load('xuexi56.pth', map_location = torch.device('cpu')))
net.eval()


while True:
    #Capture camera frame
    _, frame = cap.read()
    print('Start')

    #Got face img from frame
    face = crop_face(frame)

    #If face exists in the img, continue
    if face.all() != None:
        #Reshape face for CNN
        img = torch.Tensor(np.reshape(face/255, (1, 1, 48, 48)))

        #Show cropped face for test
        # cv2.imshow('Meshimoji', face)

        #Got Classification result
        outputs = net(img)
        _, predicted = torch.max(outputs.data, 1)

        # Test Test Test
        # print(emotions[predicted])
        cv2.putText(frame, emotions[predicted], (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    cv2.imshow('windows', frame)

    #Press q to exit
    key = cv2.waitKey() & 0xff
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
