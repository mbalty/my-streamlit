import streamlit as st
from embeddings_store import EmbeddingStore
from constants import *

def main():
    st.title("Word embeddings for text similarity")
    st.subheader('by Mihai Baltac ([@linkedin](https://www.linkedin.com/in/mihai-baltac-073108bb/))')

    """This app demonstrates word embeddings for text similarity using [sentence trasnformers](https://www.sbert.net/index.html) for creating embeddings and 
    [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for dimensinoality reduction."""

    store = emebeddings_store()

    st.sidebar.title("Input")
    st.sidebar.write(
        """ Insert word or comma separate words here to plot their relations."""
    )
    model = st.sidebar.selectbox('Model:', ST_MODELS)
    if model:
        with st.spinner(text='Loading model'):
            if store.encoder.current != model:
                store.change_model(model)
            
    placeholder = st.sidebar.empty()
    input = placeholder.text_input("Example: cat, boy, car, truck, mouse", value="")
    click_clear = st.sidebar.button('Clear All', key=1)

    if input:
        store.add_query(input)
        
    if click_clear:
        store.reset()
        input = placeholder.text_input("Example: cat, boy, car, truck, mouse", value="", key=1)
    
    image_out = store.plot(get_figure=True)
    if image_out is not None:
        st.image(image_out, use_column_width=True)
        del image_out


    st.sidebar.title("Note")
    st.sidebar.write(
        """Other notes.
        """
    )

    st.sidebar.caption(f"Streamlit version `{st.__version__}`")


@st.experimental_singleton()
def emebeddings_store():
    return EmbeddingStore(clear_memory_models=CLEAR_MEMORY_MODELS)

if __name__ == "__main__":
    main()
