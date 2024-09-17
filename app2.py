import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from io import BytesIO
import os
import gdown
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from io import BytesIO
import glob
from torch import nn
import cv2 as cv
import streamlit as st
import os
import gdown
import numpy as np
import cv2 as cv
import torch
from PIL import Image
from io import BytesIO
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb


def upsample(c_in, c_out, dropout=False):
    result = nn.Sequential()
    result.add_module('con', nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False))
    result.add_module('bat', nn.BatchNorm2d(c_out))
    if dropout:
        result.add_module('drop', nn.Dropout2d(0.5, inplace=True))
    result.add_module('relu', nn.ReLU(inplace=False))
    return result

def downsample(c_in, c_out, batchnorm=True):
    result = nn.Sequential()
    result.add_module('con', nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False))
    if batchnorm:
        result.add_module('batc', nn.BatchNorm2d(c_out))
    result.add_module('LRelu', nn.LeakyReLU(0.2, inplace=False))
    return result

class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=2, n_filters=64):
        super(Generator, self).__init__()

        layer1 = nn.Conv2d(input_nc, n_filters, kernel_size=4, stride=2, padding=1, bias=False)
        layer2 = downsample(n_filters, n_filters * 2)
        layer3 = downsample(n_filters * 2, n_filters * 4)
        layer4 = downsample(n_filters * 4, n_filters * 8)
        layer5 = downsample(n_filters * 8, n_filters * 8)
        layer6 = downsample(n_filters * 8, n_filters * 8)
        layer7 = downsample(n_filters * 8, n_filters * 8)
        layer8 = downsample(n_filters * 8, n_filters * 8)

        # Decoder
        d_inc = n_filters * 8
        dlayer8 = upsample(d_inc, n_filters * 8, dropout=True)
        dlayer7 = upsample(n_filters * 8 * 2, n_filters * 8, dropout=True)
        dlayer6 = upsample(n_filters * 8 * 2, n_filters * 8, dropout=True)
        dlayer5 = upsample(n_filters * 8 * 2, n_filters * 8)
        dlayer4 = upsample(n_filters * 8 * 2, n_filters * 4)
        dlayer3 = upsample(n_filters * 4 * 2, n_filters * 2)
        dlayer2 = upsample(n_filters * 2 * 2, n_filters)

        dlayer1 = nn.Sequential()
        dlayer1.add_module('relu', nn.ReLU(inplace=False))
        dlayer1.add_module('t_conv', nn.ConvTranspose2d(n_filters * 2, output_nc, kernel_size=4, stride=2, padding=1, bias=False))
        dlayer1.add_module('tanh', nn.Tanh())

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, input):
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        dout8 = self.dlayer8(out8)
        dout8_out7 = torch.cat([dout8, out7], 1)
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

# Fungsi untuk mengonversi Lab ke RGB tetap sama
def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)



# Fungsi unduhan file model
def download_model_if_not_exists(model_path, file_id):
    if not os.path.exists(model_path):
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, model_path, quiet=False)

# Fungsi untuk memuat model berdasarkan nama file path
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = Generator().to(device)
    net_G.load_state_dict(torch.load(model_path, map_location=device))
    net_G.eval()
    return net_G

# Dropdown untuk memilih model
model_options = {
    "Epoch 40": "1Kl5_XzzN6FT84AeJM14JT7E0cbXtA9L9",  
    "Epoch 80": "1qusiTrGxUzV6mCA4G8Sij8z-vGzyrCOX",
    "Epoch 100": "1bVKGgGjHdX4F2t3n3LiKfHyQwHj9ltTP",
    "Epoch 150": "1PMQVxvDTmLqP1DhX8xmP3K_iw_RCJsQN"
}


# Sidebar untuk memilih metode
method = st.sidebar.selectbox("Pilih Metode", ["-", "GAN (Generative Adversarial Network)", "CNN Pretrained Caffe"])


if method == "-":
    st.title("Welcome to the Batik Colorization Application Based on Deep Learning Model.")
    st.write("Please select the desired method from the sidebar to start processing images.")
    
    # Tampilkan teks dan gambar default
    st.write("""
    In this option, you can choose two image processing methods for coloring batik images.:
    
    - **GAN**: Utilizing a Generative Adversarial Network (GAN) model for batik colorization, trained on 485 images of Madura batik. This method leverages deep learning to achieve accurate and realistic colorization of batik images.
    - **CNN Pretrained Caffe**: This method utilizes a CNN model based on Pretrained Caffe, where the model has been trained with external coloring data without involving batik images.
    """)
    st.image("https://res-console.cloudinary.com/ddu9qoyjl/media_explorer_thumbnails/6c3d8c7cbaf03abf24f5e0cf9b41e227/detailed", caption="Contoh Gambar", use_column_width=True)
    st.write("*MBKM RISET 2024 Universitas Trunojoyo Madura*")


elif method == "GAN (Generative Adversarial Network)":
    selected_model_name = st.selectbox("Pilih Pretrained Model", list(model_options.keys()))
    selected_model_file_id = model_options[selected_model_name]
    model_path = f'{selected_model_name}.pth'

    # Unduh model jika belum ada
    download_model_if_not_exists(model_path, selected_model_file_id)

    # Load model yang dipilih
    net_G = load_model(model_path)

    # Definisikan device untuk pemrosesan (CPU atau GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pengunggah file (dengan multiple file upload)
    uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Crop gambar agar ukuran sama
            size = (256, 256)  # Ukuran yang diinginkan
            image_cropped = image.resize(size, Image.LANCZOS)  # Menggunakan LANCZOS sebagai alternatif

            # Pra-pemrosesan gambar
            img = np.array(image_cropped)
            img_lab = rgb2lab(img).astype("float32")
            img_lab = transforms.ToTensor()(img_lab)
            L = img_lab[[0], ...] / 50. - 1.  # Saluran warna luminance

            # Membuat tensor
            L = L.unsqueeze(0).to(device)

            # Membuat Gambar Grayscale dari saluran L
            gray_image = (L.squeeze().cpu().numpy() + 1.) * 255 / 2  # Mengonversi saluran L ke rentang [0, 255]
            gray_image = gray_image.astype(np.uint8)  # Mengubah ke uint8

            # Meneruskan melalui model
            with torch.no_grad():
                fake_color = net_G(L)
                fake_color = fake_color.detach()

            # Mengonversi Lab ke RGB
            fake_imgs = lab_to_rgb(L, fake_color)
            fake_img = fake_imgs[0]

            # Menampilkan gambar keluaran dalam satu baris
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(image_cropped, caption='Uploaded Image', use_column_width=True)

            with col2:
                st.image(gray_image, caption='Grayscale Image (L channel)', use_column_width=True, clamp=True)

            with col3:
                st.image(fake_img, caption='Colorized Image', use_column_width=True)

            # Opsi untuk mengunduh hasil
            result = Image.fromarray((fake_img * 255).astype(np.uint8))
            buf = BytesIO()
            result.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button(f"Download Result for {uploaded_file.name}", data=byte_im, file_name=f"colorized_image_{uploaded_file.name}", mime="image/jpeg")


# Model OpenCV
elif method == "CNN Pretrained Caffe":
    st.title("Image Colorization with CNN Pretrained Caffe")

    # Define paths for model files
    DIR = 'model'
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    # Paths for model files
    PROTOTXT_PATH = os.path.join(DIR, 'colorization_deploy_v2.prototxt')
    POINTS_PATH = os.path.join(DIR, 'pts_in_hull.npy')
    MODEL_PATH = os.path.join(DIR, 'colorization_release_v2.caffemodel')

    # Check if model files exist, if not download
    if not (os.path.exists(PROTOTXT_PATH) and os.path.exists(POINTS_PATH) and os.path.exists(MODEL_PATH)):
        st.write("Downloading model files...")
        # URLs for model files on Google Drive
        PROTOTXT_URL = 'https://drive.google.com/uc?id=1DZ4cFBYC3_KjOn2ayrhnk2XKHt6E54EJ'
        POINTS_URL = 'https://drive.google.com/uc?id=1Qh54l1Jhh5psiytgsv9WmJVByjpHdF8o'
        MODEL_URL = 'https://drive.google.com/uc?id=1RCb6SJN2T5tdrpPUXEx0L4GBaTtc2OcL'
        
        # Download the model files
        gdown.download(PROTOTXT_URL, PROTOTXT_PATH, quiet=False)
        gdown.download(POINTS_URL, POINTS_PATH, quiet=False)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # Load the Model
    net = cv.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    pts = np.load(POINTS_PATH)

    # Load centers for ab channel quantization used for rebalancing
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Upload images
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Read the uploaded image
            image = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv.IMREAD_COLOR)
            
            # Resize the image to 256x256 pixels
            image_resized = cv.resize(image, (256, 256))
            
            # Convert the image to grayscale
            gray_image = cv.cvtColor(image_resized, cv.COLOR_BGR2GRAY)

            # Process the image
            scaled = image_resized.astype("float32") / 255.0
            lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)
            
            resized = cv.resize(lab, (224, 224))  # Resize for model input
            L = cv.split(resized)[0]
            L -= 50
            
            st.write(f"Colorizing the image: {uploaded_file.name}...")
            net.setInput(cv.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
            
            ab = cv.resize(ab, (image_resized.shape[1], image_resized.shape[0]))
            
            L = cv.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            
            colorized = cv.cvtColor(colorized, cv.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            
            # Display images in a row
            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(image_resized, channels="BGR", caption='Uploaded Image', use_column_width=True)

            with col2:
                st.image(gray_image, channels="GRAY", caption="Grayscale Image (L channel)", use_column_width=True)

            with col3:
                st.image(colorized, channels="BGR", caption='Colorized Image', use_column_width=True)

            # Option to download the colorized image
            result_image = cv.imencode('.png', colorized)[1].tobytes()
            st.download_button(label=f"Download Colorized Image - {uploaded_file.name}", data=result_image, file_name=f"colorized_{uploaded_file.name}", mime="image/png")
