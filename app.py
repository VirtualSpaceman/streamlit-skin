import streamlit as st
import plotly.express as px
import pandas as pd

from PIL import Image
from classifier import ImageClassifier

##################
# Set page config
##################
st.set_page_config(page_title="Classificador de Imagens", page_icon="👁️", layout="wide")

###################
# Helper functions 
###################

@st.cache_resource
def load_classifier(model_card):
    image_classifier = ImageClassifier(model_card)
    return image_classifier

######################
#  Global Variables
######################

models = {
    "ResNet50": "resnet50",
    "DeiT": "deit_small_patch16_224"
}

##########
# App GUI
##########

st.title("Demo: Classificação de Imagens")

# placeholder for the header
header_placeholder = st.empty()

selected_model = st.selectbox(
    "Escolha um modelo para classificar as imagens:",
    models.keys(),
    placeholder="Escolha um modelo..."
)

if selected_model is not None:    
    # assign the header title
    header_placeholder.header(f"Classificando com {selected_model}")

    image_classifier = load_classifier(models[selected_model])

    upload = st.file_uploader(label="Carregar imagem:", type=["png", "jpg", "jpeg"])
    css = """
    <style>
        div[data-testid="stFileUploader"]>section[data-testid="stFileUploaderDropzone"]>button {{
            display: none;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>span {{
            visibility:hidden;
            font-size: 0px;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>span::after {{
            content:"{INSTRUCTIONS_TEXT}";
            visibility:visible;
            display:block;
            font-size: 16px;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>small {{
            visibility:hidden;
            font-size: 0px;
        }}
        div[data-testid="stFileUploaderDropzoneInstructions"]>div>small::before {{
            content:"{FILE_LIMITS}";
            visibility: visible;
            display:block;
            font-size: 14px;
        }}
    </style>
    """.format(
        INSTRUCTIONS_TEXT="Arraste e solte a imagem aqui ou clique aqui para carregar.",
        FILE_LIMITS="Limite de 200MB por arquivo • PNG, JPG, JPEG",
    )
    st.markdown(css, unsafe_allow_html=True)

    if upload:
        img = Image.open(upload).convert("RGB")

        scores = image_classifier.predict(img)
        scores_df = pd.DataFrame.from_dict(scores)
        prediction = scores_df['score'].idxmax()

        top_label = scores_df.loc[prediction, 'label']
        top_prob = scores_df.loc[prediction, 'score']

        top_scores = scores_df.sort_values(by="score").tail(2)
        top_scores['score'] = top_scores['score'] * 100

        top_scores['Probabilidade'] = top_scores['score']
        top_scores['Categoria'] = top_scores['label']

        fig = px.bar(top_scores, x="Probabilidade", y="Categoria", orientation='h', title="Previsão")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Classificação:")
            st.markdown(f"Categoria \"**{top_label}**\" com {top_prob*100:.2f}% de probabilidade.")
            st.plotly_chart(fig, theme="streamlit")

        with col2:
            st.subheader("Imagem:")
            st.image(img, caption=f"Uploaded image: {upload.name}")