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

    prediction = float(model.predict(img_array)[0][0])

    # 모델 출력값 해석 확인용
    st.subheader("예측 결과")
    st.write(f"모델 원본 출력값: {prediction:.4f}")

    # 현재 모델이 fake 확률을 출력한다고 가정
    fake_probability = prediction
    real_probability = 1 - prediction

    # 너무 Real로만 나오는 문제를 완화하기 위해 threshold 조정
    threshold = 0.20

    if fake_probability >= threshold:
        st.error("예측: 가짜")
    else:
        st.success("예측: 진짜")

    st.write(f"가짜 확률: {fake_probability:.4f}")
    st.write(f"진짜 확률: {real_probability:.4f}")
    st.write(f"판정 기준값: {threshold}")

    st.caption(
        "※ 이 모델은 제한된 데이터로 학습되어 이미지 품질, 얼굴 크롭, 조명, 압축 정도에 따라 예측이 달라질 수 있습니다."
    )

else:
    st.info("이미지를 업로드하면 예측이 시작됩니다.")