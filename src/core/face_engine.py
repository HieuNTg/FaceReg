import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import streamlit as st
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(classify=False, pretrained="vggface2")
    model.eval()
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.4, 0.5, 0.5],
        factor=0.709,
        post_process=True,
        select_largest=True,
        device=device,
    )
    return model, mtcnn


@st.cache_resource
def load_data():
    model_weights = torch.load(MODELS_DIR / "vector_face.pth", weights_only=True)
    usernames = np.load(MODELS_DIR / "name_face.npy", allow_pickle=True)
    return model_weights, usernames


def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum(torch.pow(tensor1 - tensor2, 2)))


class FaceEngine:
    def __init__(self):
        self.model, self.mtcnn = load_models()
        self.model_weights, self.usernames = load_data()

    def process(self, frame):
        filename_crop = self.mtcnn(frame)
        box, _ = self.mtcnn.detect(frame)

        if filename_crop is not None:
            box = box[0]
            result = self.model(filename_crop.reshape(1, 3, 160, 160))
            prob = []
            for i in range(len(self.usernames)):
                prob.append(euclidean_distance(result[0], self.model_weights[i]))
            prob = np.array([ele.detach().numpy() for ele in prob], dtype=float)
            min_distance = np.min(prob)

            bbox = list(map(int, box.tolist()))
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + w, bbox[1] + h), (255, 255, 0), 2)

            confidence = round(prob[np.argmin(prob)] % 1, 2)

            if min_distance > 2.0:
                text = f"UNK    {confidence}"
            else:
                text = f"{self.usernames[np.argmin(prob)]}    {confidence}"

            cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        return frame

    def train_from_frames(self, frames, username):
        embs = []
        annotated_frames = []
        for frame in frames:
            filename_crop = self.mtcnn(frame)
            box, _ = self.mtcnn.detect(frame)
            if filename_crop is not None:
                box = box[0]
                bbox = list(map(int, box.tolist()))
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + w, bbox[1] + h), (255, 255, 0), 2)
                result = self.model(filename_crop.reshape(1, 3, 160, 160))
                embs.append(result)
            annotated_frames.append(frame)

        if len(embs) == 0:
            return False, annotated_frames

        embedding = torch.cat(embs).mean(0, keepdim=True)
        my_tensors = [item.unsqueeze(0) for item in self.model_weights]
        my_tensors.append(embedding)
        my_tensors = torch.cat(my_tensors)

        self.usernames = np.append(self.usernames, username)
        torch.save(my_tensors, MODELS_DIR / "vector_face.pth")
        np.save(MODELS_DIR / "name_face.npy", self.usernames)
        self.model_weights = my_tensors

        return True, annotated_frames
