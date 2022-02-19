from embedding import EmbeddingStore
import numpy as np
import os
import streamlit as st
import sys
import urllib



def main():
    st.title("Word embeddings for text similarity")
    st.subheader('by Mihai Baltac ([@linkedin](https://www.linkedin.com/in/mihai-baltac-073108bb/))')

    """This app demonstrates word embeddings for text similarity using [sentence trasnformers](https://www.sbert.net/index.html) for creating embeddings and 
    [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for dimensinoality reduction."""


    with st.spinner(text='Loading models'):
        store = load_store()
        st.success('Done')

    st.sidebar.title("Input")
    st.sidebar.write(
        """ Insert word or comma separate words here to plot their relations."""
    )
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
    
    

    st.sidebar.title("Note")
    st.sidebar.write(
        """Other notes.
        """
    )

    st.sidebar.caption(f"Streamlit version `{st.__version__}`")



@st.experimental_singleton()
def load_store():
    return EmbeddingStore()


if __name__ == "__main__":
    main()
