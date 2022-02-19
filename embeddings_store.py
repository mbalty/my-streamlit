from operator import mod
from webbrowser import get
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from constants import *



class EmbeddingEncoder:
    transformers: Dict[str, SentenceTransformer]
    
    # todo: will be an interface implemented by multiple types of encoders
    def __init__(self, load=False, clear_memory=False):
        self.transformers = dict()
        self.current = None
        self.clear_memory = clear_memory
        if load:
            self.load()
    
    def load(self, model_name=DEFAULT):
        if model_name not in self.transformers:
            if self.clear_memory and self.current is not None:
                del self.transformers[self.current]
            self.transformers[model_name] = SentenceTransformer(model_name)
        self.current = model_name

    def encode(self, query):
        return self.transformers[self.current].encode(query)
    

class EmbeddingStore:
    def __init__(self, clear_memory_models=False):
        self.embeddings = []
        self.compressed_points = []
        self.queries = []
        self.encoder = EmbeddingEncoder(clear_memory=clear_memory_models)
        self.pca = PCA(n_components=2)
    
    def add_query(self, query):
        if isinstance(query, list):
            self._add_query_list(query)
        elif isinstance(query, str):
            self._add_query_list(query.split(","))
        else:
            raise Exception("Bad query type")
        
        
    def _add_query_list(self, query_list):
        for query in query_list:
            if query not in self.queries:
                self.queries.append(query)
                self.embeddings.append(self.encoder.encode(query))
    
        if len(self.embeddings)> 1:
            self.compressed_points = self.pca.fit_transform(self.embeddings + [self.encoder.encode("")])[:-1]        
    
    def plot(self, get_figure=False):
        points = [[0,0]]
        if len(self.compressed_points)>1:
            points = self.compressed_points
            
        if get_figure:
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()
        else:
            fig, ax = plt.subplots()
        ax.axis('off')
        ax.scatter(*zip(*points))

        for i, txt in enumerate(self.queries):
            ax.annotate(txt, points[i])
            
        if get_figure and len(self.queries)>0:
            canvas.draw()       # draw the canvas, cache the renderer
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return image_from_plot    
    
    def change_model(self, model_name):
        print("model changing to ", model_name)
        self.encoder.load(model_name)
        existing_queries = self.queries
        self.reset()
        self.add_query(existing_queries)
    
    def reset(self):
        print ("reseting queries")
        self.embeddings = []
        self.compressed_points = []
        self.queries = []