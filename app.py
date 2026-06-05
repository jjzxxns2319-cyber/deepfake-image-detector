import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="Deepfake Detection", page_icon="🧠")

st.title("딥페이크 이미지 감지")
st.write("얼굴 이미지를 업로드하고 진짜인지 가짜인지 확인하세요.")

@st.cache_resource
def load_cnn_model():
    if not os.path.exists("cnn_model.h5"):
        gdown.download(
            "https://drive.google.com/uc?id=1OeIqW5Dqar2SNGktBWYerp_YSRt1p3LH",
            "cnn_model.h5",
            quiet=False
        )
    return tf.keras.models.load_model("cnn_model.h5")

model = load_cnn_model()

uploaded_file = st.file_uploader(
    "이미지 업로드하기",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    fake_probability = float(model.predict(img_array)[0][0])

    st.subheader("예측 결과")
    st.write(f"예측 확률 (Fake): {fake_probability:.4f}")

    if fake_probability >= 0.2:
        st.error("예측: 가짜 (Fake)")
    else:
        st.success("예측: 진짜 (Real)")

else:
    st.info("이미지를 업로드하면 예측이 시작됩니다.")