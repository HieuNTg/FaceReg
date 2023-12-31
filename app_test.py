import cv2
import numpy as np
import torch
import streamlit as st
import io
import tempfile
import base64
import av
import threading
import os

from twilio.rest import Client
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

new_title = "FaceReg"
st.set_page_config(page_title=new_title)

def get_base64(gif_file):
    with open(gif_file, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode("utf-8")

def set_background(gif_file):
    base64_str = get_base64(gif_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/gif;base64,{base64_str}");
    background-size: cover;
    background-attachment: fixed;
    margin: 0;
    padding: 0;
    height: 100vh;
    overflow: hidden;
    opacity: 0.9;
    }}


    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
def set_background_sidebar(gif_file):
    base64_str = get_base64(gif_file)
    page_bg_img = f'''
    <style>
    .st-emotion-cache-1cypcdb {{
    background-image: url("data:image/jpg;base64,{base64_str}");
    background-size: cover;
    margin: 0;
    padding: 0;
    height: 100vh;
    overflow: hidden;
    opacity: 0.9;
    object-fit: contain;
    max-width: 100%;
    }}


    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(
        classify=False,
        pretrained='vggface2',
    )
    model.eval()
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.4, 0.5, 0.5], factor=0.709, post_process=True,
        select_largest=True,
        device=device
    )
    return model, mtcnn

@st.cache_resource
def load_data():
    model_weights = torch.load('vector_face.pth')
    usernames = np.load('name_face.npy')
    return model_weights, usernames

def euclidean_distance(tensor1, tensor2):
    
     # Tính tổng bình phương của chênh lệch giữa các giá trị của 2 tensor
    squared_difference = torch.pow(tensor1 - tensor2, 2)

    # Tính tổng bình phương của các giá trị của 2 tensor
    squared_norm = torch.sum(squared_difference)

    # Trả về khoảng cách Euclidian
    return torch.sqrt(squared_norm)

def process(frame):
    # Gọi hàm detect và crop khuôn mặt
    filename_crop = mtcnn(frame)
    box, _ = mtcnn.detect(frame)

    # Duyệt qua danh sách vector khuôn mặt và so sánh
    if filename_crop is not None:
        box = box[0]
        result = model(filename_crop.reshape(1, 3, 160, 160))
        prob = []
        for i in range(len(usernames)):
            prob.append(euclidean_distance(result[0], model_weights[i]))
        prob = np.array([ele.detach().numpy() for ele in prob], dtype=float)
        min_distance = np.min(prob)

        # Hiển thị khung hình và hộp giới hạn
        bbox = list(map(int, box.tolist()))
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + w, bbox[1] + h), (255, 255, 0), 2)
        decimal_part = str(prob[np.argmin(prob)]).split('.')[1]
        pro = round(float('0.' + decimal_part), 2)
        if min_distance > 2.0:
            text = "UNK" + 2 * "  " + str(pro)
        else:
            text = usernames[np.argmin(prob)] + 2 * "  " + str(pro)
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    # Hiển thị khung hình với văn bản và hộp giới hạn
    return frame

def ImageRecognition():
    st.title("Demo FaceRecognition")
    set_background('img_reg.gif')
    
    uploaded_file = st.file_uploader("Choose a image", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img)
        img = process(img)
        st.image(img)
        
        
def VideoRecognition():  
    st.title("Demo FaceRecognition")
    set_background('img_reg.gif')
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
    col1, col2 = st.columns(2)
    temporary_location = False
    if uploaded_file is not None:
        with col1:
            text = st.empty()
        text.write("Đang trong quá trình xử lí...")
        g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        temporary_location = "testout_simple.mp4"

        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
        # close file
        out.close()
        output_video_path = 'result.mp4'
        if temporary_location:
            
            sf = st.empty()
            video_stream = cv2.VideoCapture(temporary_location)
            # Lấy thông tin về video gốc
            frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_stream.get(cv2.CAP_PROP_FPS))

            # Tạo đối tượng VideoWriter để ghi video đã xử lí
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video đầu ra (ở đây là MP4)
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            while video_stream:
                
                ret, frame = video_stream.read()
                if not ret:
                    break
                frame = process(frame)
                sf.image(frame, channels='BGR')
                out.write(frame)
            # Giải phóng tài nguyên
            video_stream.release()
            out.release()
            cv2.destroyAllWindows()
            
            # st.video(output_video_path)
        text.write("Tải video bằng cách bấm download phía dưới:")
        with open("result.mp4", "rb") as file:
            with col2:
                st.download_button(
                    label = "Download",
                    data = file,
                    file_name="result.mp4",
                    mime="video/mp4")

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

token = client.tokens.create()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": token.ice_servers}
)

class VideoProcessor():
    def __init__(self):
        self.frame_count = 0  # Initialize frame counter

    def recv(self, frame):
        self.frame_count += 1  # Increment frame counter

        if self.frame_count % 2 == 0:  
            img = frame.to_ndarray(format="bgr24")
            img = process(img)  # Apply your processing logic here
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            return frame

def Real_timeRecognition( usernames, model_weights, ec, cap):  
    st.title("Demo FaceRecognition RealTime")
    set_background('img_reg.gif')
    
    webrtc_ctx = webrtc_streamer(
        key="WHO",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True)

    
def Video_train():
    global usernames 
    global model_weights
    st.title("Train by UpLoad Video")
    set_background('img_reg.gif')
    with st.form('form1'):
        username = st.text_input('User name:',  max_chars=100)
        submit = st.form_submit_button('Submit')
    if username:
        # Tải lên tệp video
        st.info('Vui lòng upload video quay đủ góc cạnh của khuôn mặt !')
        video_file = st.file_uploader("Chọn tệp video", type=["mp4"])
        
        if video_file is not None:
            embs = [] 
            usernames = np.append(usernames, username)
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(video_file.read())
            vidcap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            t = 0
            while vidcap.isOpened():
                # Đọc khung hình từ camera
                ret, frame = vidcap.read()
                if ret:
                    filename_crop = mtcnn(frame)
                    box, _ = mtcnn.detect(frame)
                    if filename_crop is not None:
                        box = box[0]
                        # for display image and box  
                        bbox = list(map(int, box.tolist()))
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + w, bbox[1] + h), (255, 255, 0), 2)
                        result = model(filename_crop.reshape(1, 3, 160, 160))
                        embs.append(result)
                    stframe.image(frame, channels='BRG')
                    t += 1

                else:
                    break    
                if t == 100:
                    break
            embedding = torch.cat(embs).mean(0, keepdim=True)
            my_tensors = []
            for item in model_weights:
                item = item.unsqueeze(0)
                my_tensors.append(item)
            my_tensors.append(embedding)
            my_tensors = torch.cat(my_tensors)
            # st.write(len(my_tensors), len(usernames))
            torch.save(my_tensors, "./vector_face.pth")
            np.save('./name_face.npy', usernames)
            st.info("Quá trình train đã thành công")


lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    return frame

def RealTime_train( cap):
    global usernames 
    global model_weights
    st.title("Train by Realtime")
    set_background('img_reg.gif')
    submit = False
    with st.form('form1'):
        username = st.text_input('User name:',  max_chars=100)
        usernames = np.append(usernames, username)
        submit = st.form_submit_button('Submit')
    if username:
        embs = []
        ctx = webrtc_streamer(key="example", 
                              rtc_configuration=RTC_CONFIGURATION,
                              video_frame_callback=video_frame_callback)
        
        while ctx.state.playing:
            with lock:
                frame = img_container["img"]
            if frame is None:
                continue
        # if frame is not None:
        
            filename_crop = mtcnn(frame)
            box, _ = mtcnn.detect(frame)
            if filename_crop is not None:
                box = box[0]
                # for display image and box  
                bbox = list(map(int, box.tolist()))
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + w, bbox[1] + h), (255, 255, 0), 2)
                result = model(filename_crop.reshape(1, 3, 160, 160))
                embs.append(result)
        embedding = None
        if len(embs) > 0:
            embedding = torch.cat(embs).mean(0, keepdim=True)
        if embedding is not None:
            my_tensors = []
            for item in model_weights:
                item = item.unsqueeze(0)
                my_tensors.append(item)
            my_tensors.append(embedding)
            my_tensors = torch.cat(my_tensors)
            torch.save(my_tensors, "./vector_face.pth")
            np.save('./name_face.npy', usernames)
            st.info("Quá trình train đã thành công")
                
if __name__ == "__main__":
    
    model_weights, usernames = load_data()
    model, mtcnn = load_models()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Using object notation
    add_selectbox = st.sidebar.selectbox(
        "What do you want to do?",
        ("Recognition", "Train a new face")
    )
    set_background_sidebar("download.jpg")
    st.sidebar.markdown(
    """
    <style>


    .st-bn {
        color: red;
    }
    
    .st-az {
        font-family: "Georgia", sans-serif;
    }
    
    .st-bb {
        background-color: transparent;
        border: 2px solid blue;
    }

    .st-emotion-cache-15wihvi {
        background: #7CFC00;
    }
    
    .st-emotion-cache-fblp2m {
        color: red;
    }
    
    .st-b6 {
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True)
    st.markdown(
    """
    <style>


    .st-emotion-cache-1avcm0n {
        height: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True)
    
    st.markdown(
    """
    <style>
    .st-emotion-cache-10trblm {
        font-family: "Georgia", sans-serif;
        text-align: center;
        color : blue;
    }

    .st-emotion-cache-q8sbsg {
        font-family: "Georgia", sans-serif;
        color : blue;
        
    }
    .st-emotion-cache-q8sbsg p {
        font-family: "Georgia", sans-serif;
        font-size: 16px;
    }
    
    .st-emotion-cache-1erivf3 {
        font-family: "Georgia", sans-serif;
        background-color: transparent;
        color: red;
    }
    
    .st-emotion-cache-noeb3a {
        color: red;
    }
    .st-emotion-cache-7oyrr6 {
        font-family: "Georgia", sans-serif;
        color: red;
    }
    
    .st-emotion-cache-19rxjzo {
        border: 2px solid blue;  
    }
    
    .st-emotion-cache-1uixxvy {
        color: red;
    }
    
    .st-emotion-cache-vskyf7 {
        color: red;
    }
    
    img {
        border: 2px solid blue; 
    }
    
    .st-emotion-cache-x78sv8 {
        font-family: "Georgia", sans-serif;
        color: red;
    }
    
    .st-emotion-cache-1xw8zd0 {
        border: 2px solid blue;
        border-radius: 0.5rem;
    }
    
    .st-f2 {
        color: yellow;
    }
    .st-f8 {
        color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True)
    
    if add_selectbox == "Recognition":
        with st.sidebar:
            add_radio = st.radio("What do you want to choice?",
                            ("Image", "Video", "Real-Time"))
        if (add_radio=="Image"):
            ImageRecognition()
        elif (add_radio=="Video"):
            VideoRecognition()
        else:
            Real_timeRecognition(usernames, model_weights, euclidean_distance, cap)
    else:
        with st.sidebar:
            add_radio = st.radio("What do you want to choice?",
                            ("Real-time", "Video"))
        if (add_radio=="Video"):
            Video_train()
        else:
            RealTime_train(cap)
        


    
    