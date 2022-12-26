import streamlit as st
import numpy as np
from utils import build_transform, load_image_from_url
from model import ScoringNet

st.title("Test page for _ScoringNet_ :100:")
svr_path = "LinearSVR_clip.joblib"
transform = build_transform()
model = ScoringNet(svr_path=svr_path)

url = st.text_input(label="type an url of a test image", value="https://bit.ly/3Z08iuC")
img = load_image_from_url(url=url, div=5)

with st.container():
    st.write("This is the image from your url :camera_with_flash:")
    st.image(img, channels="RGB")

img = img.convert("RGB")
input_tensor = transform(image=np.array(img))["image"].unsqueeze(0)
logit = model(input_tensor)
output = logit.item()

st.subheader("Score is")
st.subheader(":green[{}]".format(round(output, 3)))
