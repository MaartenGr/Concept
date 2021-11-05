import joblib
import hdbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Mapping, Tuple
from PIL import Image
from umap import UMAP
from scipy.sparse.csr import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from concept._mmr import mmr
from concept._ctfidf import ClassTFIDF
from concept._visualization import get_concat_tile_resize


class ConceptModel:
    """ Concept is a technique that leverages CLIP and BERTopic-based
    techniques to perform Concept Modeling on images.

    Since topics are part of conversations and text, they do not
    represent the context of images well. Therefore, these clusters of images are
    referred to as 'Concepts' instead of the traditional 'Topics'.

    Thus, Concept Modeling takes inspiration from topic modeling techniques
    to cluster images, find common concepts and model them both visually
    using images and textually using topic representations.

     Usage:
    ```python
    from concept import ConceptModel

    concept_model = ConceptModel()
    concept_clusters = concept_model.fit_transform(images)
    ```
    """
    def __init__(self,
                 min_concept_size: int = 30,
                 diversity: float = 0.3,
                 embedding_model: str = "clip-ViT-B-32",
                 vectorizer_model: CountVectorizer = None,
                 umap_model: UMAP = None,
                 hdbscan_model: hdbscan.HDBSCAN = None,
                 ctfidf: bool = False):
        """ Concept Model Initialization

        Arguments:
            min_concept_size: The minimum size of concepts. Increasing this value will lead
                              to a lower number of concept clusters.
            diversity: How diverse the images within a concept are.
                       Values between 0 and 1 with 0 being not diverse at all
                       and 1 being most diverse.
            embedding_model: The CLIP model to use. Current options include:
                    * clip-ViT-B-32
                    * clip-ViT-B-32-multilingual-v1
            vectorizer_model: Pass in a CountVectorizer instead of the default
            umap_model: Pass in a UMAP model to be used instead of the default
            hdbscan_model: Pass in a hdbscan.HDBSCAN model to be used instead of the default
            ctfidf: Whether to use c-TF-IDF to create the textual concept representation
        """
        self.diversity = diversity
        self.min_concept_size = min_concept_size

        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Vectorizer
        self.vectorizer_model = vectorizer_model or CountVectorizer()

        # UMAP
        self.umap_model = umap_model or UMAP(n_neighbors=15,
                                             n_components=5,
                                             min_dist=0.0,
                                             metric='cosine')

        # HDBSCAN
        self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(min_cluster_size=self.min_concept_size,
                                                              metric='euclidean',
                                                              cluster_selection_method='eom',
                                                              prediction_data=True)

        self.frequency = None
        self.topics = None
        self.cluster_embeddings = None
        self.ctfidf = ctfidf

    def fit_transform(self,
                      images: List[str],
                      docs: List[str] = None,
                      image_names: List[str] = None,
                      image_embeddings: np.ndarray = None) -> List[int]:
        """ Fit the model on a collection of images and return concepts

        Arguments:
            images: A list of paths to each image
            docs: The documents from which to extract textual concept representation
            image_names: The names of the images for easier
                         reading of concept clusters
            image_embeddings: Pre-trained image embeddings to use
                              instead of generating them in Concept

        Returns:
            predictions: Concept prediction for each image

        Usage:

        ```python
        from concept import ConceptModel
        concept_model = ConceptModel()
        concepts = concept_model.fit_transform(images)
        ```
        """

        # Calculate image embeddings if not already generated
        if image_embeddings is None:
            image_embeddings = self._embed_images(images)

        # Reduce dimensionality and cluster images into concepts
        reduced_embeddings = self._reduce_dimensionality(image_embeddings)
        predictions = self._cluster_embeddings(reduced_embeddings)

        # Extract representative images through exemplars
        representative_images = self._extract_exemplars(image_names)
        exemplar_embeddings = self._extract_cluster_embeddings(image_embeddings,
                                                               representative_images)
        selected_exemplars = self._extract_exemplar_subset(exemplar_embeddings,
                                                           representative_images)

        # Create collective representation of images
        self._cluster_representation(images, selected_exemplars)

        # Find the best words for each concept cluster
        if docs is not None:
            if self.ctfidf:
                self._extract_ctfidf_representation(docs, image_embeddings)
            else:
                self._extract_textual_representation(docs)

        return predictions

    def fit(self,
            images: List[str],
            image_names: List[str] = None,
            image_embeddings: np.ndarray = None):
        """ Fit the model on a collection of images and return concepts

        Arguments:
            images: A list of paths to each image
            image_names: The names of the images for easier
                         reading of concept clusters
            image_embeddings: Pre-trained image embeddings to use
                              instead of generating them in Concept

        Usage:

        ```python
        from concept import ConceptModel
        concept_model = ConceptModel()
        concepts = concept_model.fit(images)
        ```
        """
        self.fit_transform(images, image_names=image_names, image_embeddings=image_embeddings)
        return self

    def transform(self, images, image_embeddings=None):
        """ After having fit a model, use transform to predict new instances

        Arguments:
            images: A single images or a list of images to predict
            image_embeddings: Pre-trained image embeddings. These can be used
                              instead of the sentence-transformer model.
        Returns:
            predictions: Concept predictions for each image

        Usage:
        ```python
        concept_model = ConceptModel()
        concepts = concept_model.fit(images)
        new_concepts = concept_model.transform(new_images)
        ```
        """
        if image_embeddings is not None:
            image_embeddings = self._embed_images(images)

        umap_embeddings = self.umap_model.transform(image_embeddings)
        predictions, _ = hdbscan.approximate_predict(self.hdbscan_model, umap_embeddings)
        return predictions

    def _embed_images(self,
                      images: List[str]) -> np.ndarray:
        """ Embed the images

        Not entirely sure why but the RAM ramps up
        if I do not close the images between batches.
        So I make a copy out of those and simply
        close them in between without touching the original
        images.

        Arguments:
            images: A list of paths to each image

        Returns:
            embeddings: The image embeddings
        """
        # Prepare images
        batch_size = 128
        nr_iterations = int(np.ceil(len(images) / batch_size))

        # Embed images per batch
        embeddings = []
        for i in tqdm(range(nr_iterations)):
            start_index = i * batch_size
            end_index = (i * batch_size) + batch_size

            images_to_embed = [Image.open(filepath) for filepath in images[start_index:end_index]]
            img_emb = self.embedding_model.encode(images_to_embed, show_progress_bar=False)
            embeddings.extend(img_emb.tolist())

            # Close images
            for image in images_to_embed:
                image.close()

        return np.array(embeddings)

    def _reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """ Reduce dimensionality of embeddings using UMAP

        Arguments:
            embeddings: The extracted embeddings using the sentence transformer module.

        Returns:
            umap_embeddings: The reduced embeddings
        """
        self.umap_model.fit(embeddings)
        umap_embeddings = self.umap_model.transform(embeddings)
        return umap_embeddings

    def _cluster_embeddings(self,
                            embeddings: np.ndarray) -> List[int]:
        """ Cluster UMAP embeddings with HDBSCAN

        Arguments:
            embeddings: The reduced sentence embeddings

        Returns:
            predicted_clusters: The predicted concept cluster for each image
        """
        self.hdbscan_model.fit(embeddings)
        self.cluster_labels = sorted(list(set(self.hdbscan_model.labels_)))
        predicted_clusters = list(self.hdbscan_model.labels_)

        self.frequency = (
            pd.DataFrame({"Cluster": predicted_clusters, "Count": predicted_clusters})
              .groupby("Cluster")
              .count()
              .drop(-1)
              .sort_values("Count", ascending=False)
        )
        return predicted_clusters

    def _extract_exemplars(self,
                           image_names: List[str] = None) -> Mapping[str, Mapping[str, List[int]]]:
        """ Save the most representative images per concept

          The most representative images are extracted by taking
          the exemplars from the HDBSCAN-generated clusters.

          Full instructions can be found here:
              https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html

          Arguments:
              image_names: The name of images if supplied otherwise use indices
        """
        if not image_names:
            image_names = [i for i in range(len(self.hdbscan_model.labels_))]

        # Prepare the condensed tree
        condensed_tree = self.hdbscan_model.condensed_tree_
        raw_tree = condensed_tree._raw_tree
        clusters = sorted(condensed_tree._select_clusters())
        cluster_tree = raw_tree[raw_tree['child_size'] > 1]

        #  Find the points with maximum lambda value in each leaf
        representative_images = {}
        for cluster in self.cluster_labels:
            if cluster != -1:
                leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, clusters[cluster])

                exemplars = np.array([])
                for leaf in leaves:
                    max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
                    points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                               (raw_tree['lambda_val'] == max_lambda)]
                    exemplars = np.hstack((exemplars, points))

                representative_images[cluster] = {"Indices": [int(index) for index in exemplars],
                                                  "Names": [image_names[int(index)] for index in exemplars]}

        return representative_images

    def _extract_cluster_embeddings(self,
                                    image_embeddings: np.ndarray,
                                    representative_images: Mapping[str,
                                                                   Mapping[str,
                                                                           List[int]]]) -> Mapping[str, np.ndarray]:
        """ Create a concept cluster embedding for each concept cluster by
        averaging the exemplar embeddings for each concept cluster.

        Arguments:
            image_embeddings: All image embeddings
            representative_images: The representative images per concept cluster

        Updates:
            cluster_embeddings: The embeddings for each concept cluster

        Returns:
            exemplar_embeddings: The embeddings for each exemplar image
        """
        exemplar_embeddings = {}
        cluster_embeddings = []
        for label in self.cluster_labels[1:]:
            embeddings = image_embeddings[np.array([index for index in
                                                    representative_images[label]["Indices"]])]
            cluster_embedding = np.mean(embeddings, axis=0).reshape(1, -1)

            exemplar_embeddings[label] = embeddings
            cluster_embeddings.append(cluster_embedding)

        self.cluster_embeddings = cluster_embeddings

        return exemplar_embeddings

    def _extract_exemplar_subset(self,
                                 exemplar_embeddings: Mapping[str, np.ndarray],
                                 representative_images: Mapping[str, Mapping[str,
                                                                             List[int]]]) -> Mapping[str, List[int]]:
        """ Use MMR to filter out images in the exemplar set

        Arguments:
            exemplar_embeddings: The embeddings for each exemplar image
            representative_images: The representative images per concept cluster

        Returns:
            selected_exemplars: A selection (8) of exemplar images for each concept cluster
        """

        selected_exemplars = {cluster: mmr(self.cluster_embeddings[cluster],
                                           exemplar_embeddings[cluster],
                                           representative_images[cluster]["Indices"],
                                           diversity=self.diversity,
                                           top_n=8)
                              for index, cluster in enumerate(self.cluster_labels[1:])}

        return selected_exemplars

    def _cluster_representation(self,
                                images: List[str],
                                selected_exemplars: Mapping[str, List[int]]):
        """ Cluster exemplars into a single image per concept cluster

        Arguments:
            images: A list of paths to each image
            selected_exemplars: A selection of exemplar images for each concept cluster
        """
        # Find indices of exemplars per cluster
        sliced_exemplars = {cluster: [[j for j in selected_exemplars[cluster][i:i + 3]]
                                      for i in range(0, len(selected_exemplars[cluster]), 3)]
                            for cluster in self.cluster_labels[1:]}
        
        # combine exemplars into a single image
        cluster_images = {}
        for cluster in self.cluster_labels[1:]:
            images_to_cluster = [[Image.open(images[index]) for index in sub_indices] for sub_indices in sliced_exemplars[cluster]]
            cluster_image = get_concat_tile_resize(images_to_cluster)
            cluster_images[cluster] = cluster_image

            # Make sure to properly close images
            for image in images_to_cluster:
                image.close()

        self.cluster_images = cluster_images

    def _extract_textual_representation(self,
                                        docs: List[str]):
        """ Extract textual representation of concepts by comparing with documents

        Arguments:
            docs: A list of documents from which to extract words/tokens
        """

        # Extract vocabulary from the documents
        self.vectorizer_model.fit(docs)
        words = self.vectorizer_model.get_feature_names()

        # Embed the documents and extract similarity between concept clusters and words
        text_embeddings = self.embedding_model.encode(words, show_progress_bar=True)
        sim_matrix = cosine_similarity(np.array(self.cluster_embeddings)[:, 0, :], text_embeddings)

        # Extract most similar words for each concept cluster
        topics = {}
        for index in range(sim_matrix.shape[0]):
            indices = np.argpartition(sim_matrix[index], -5)[-5:]
            topics[index] = ", ".join([words[index] for index in indices])

        self.topics = topics

    def _extract_ctfidf_representation(self,
                                       docs: List[str],
                                       image_embeddings: np.ndarray):
        """ Extract textual representation through c-TF-IDF

        For each image, generate 10 related words. Then, combine the words
        of each image in a cluster and run c-TF-IDF over all clusters.

        Arguments:
            docs: The documents from which to extract the words
            image_embeddings: All image embeddings
        """
        # Extract vocabulary from the documents
        self.vectorizer_model.fit(docs)
        words = self.vectorizer_model.get_feature_names()

        # Embed the documents and extract similarity between concept clusters and words
        text_embeddings = self.embedding_model.encode(words, show_progress_bar=True)
        sim_matrix = cosine_similarity(image_embeddings, text_embeddings)

        # Extract most similar words for each concept cluster
        image_words = []
        for index in tqdm(range(sim_matrix.shape[0])):
            indices = np.argpartition(sim_matrix[index], -20)[-20:]
            selected_words = " ".join([words[index] for index in indices])
            image_words.append(selected_words)

        df = pd.DataFrame({"Words": image_words, "Concept": list(self.hdbscan_model.labels_)})
        documents_per_concept = df.groupby(['Concept'], as_index=False).agg({'Words': ' '.join})

        # Extract c-TF-IDF representation
        m = len(df)
        documents = documents_per_concept.Words.tolist()
        self.vectorizer_model.fit(documents)
        selected_words = self.vectorizer_model.get_feature_names()
        X = self.vectorizer_model.transform(documents)

        transformer = ClassTFIDF().fit(X, n_samples=m, multiplier=None)
        c_tf_idf = transformer.transform(X)
        labels = sorted(list(set(list(self.hdbscan_model.labels_))))

        # Get the top 10 indices and values per row in a sparse c-TF-IDF matrix
        indices = self._top_n_idx_sparse(c_tf_idf, 5)
        scores = self._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {label: [(selected_words[word_index], score)
                          if word_index and score > 0
                          else ("", 0.00001)
                          for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                          ]
                  for index, label in enumerate(labels)}
        self.topics = {label: ", ".join([word for word, _ in values]) for label, values in topics.items()}

    def find_concepts(self, search_term: str) -> List[Tuple[int, float]]:
        """ Based on a search term, find the top 5 related concepts

        Arguments:
            search_term: The search term to search for

        Returns:
            results: The top 5 related concepts with their similarity scores

        Usage:

        ```python
        results = concept_model.find_concepts(search_term="dog")
        ```
        """
        embedding = self.embedding_model.encode(search_term)
        sim_matrix = cosine_similarity(embedding.reshape(1, -1), np.array(self.cluster_embeddings)[:, 0, :])
        related_concepts = np.argsort(sim_matrix)[0][::-1][:5]
        vals = list(np.sort(sim_matrix)[0][::-1][:5])

        results = [(concept, val) for concept, val in zip(related_concepts, vals)]
        return results

    def visualize_concepts(self,
                           top_n: int = 9,
                           concepts: List[int] = None,
                           figsize: Tuple[int, int] = (20, 15)):
        """ Visualize concepts using merged exemplars

        Arguments:
            top_n: The top_n concepts to visualize
            concepts: The concept clusters to visualize
            figsize: The size of the figure
        """
        if not concepts:
            concepts = [self.frequency.index[index] for index in range(top_n)]
            images = [self.cluster_images[index] for index in concepts]
        else:
            images = [self.cluster_images[index] for index in concepts]

        nr_columns = 3 if len(images) >= 3 else len(images)
        nr_rows = int(np.ceil(len(concepts) / nr_columns))

        fig, axs = plt.subplots(nr_rows, nr_columns, figsize=figsize)

        # visualize multiple concepts
        if len(images) > 1:
            axs = axs.flatten()
            for index, ax in enumerate(axs):
                if index < len(images):
                    ax.imshow(images[index])
                    if self.topics:
                        title = f"Concept {concepts[index]}: \n{self.topics[concepts[index]]}"
                    else:
                        title = f"Concept {concepts[index]}"
                    ax.set_title(title)
                ax.axis('off')

        # visualize a single concept
        else:
            axs.imshow(images[0])
            if self.topics:
                title = f"Concept {concepts[0]}: \n{self.topics[concepts[0]]}"
            else:
                title = f"Concept {concepts[0]}"
            axs.set_title(title)
            axs.axis('off')
        return fig

    def save(self,
             path: str) -> None:
        """ Saves the model to the specified path

        Arguments:
            path: the location and name of the file you want to save

        Usage:
        ```python
        concept_model.save("my_model")
        ```
        """
        with open(path, 'wb') as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls,
             path: str):
        """ Loads the model from the specified path

        Arguments:
            path: the location and name of the ConceptModel file you want to load

        Usage:
        ```python
        ConceptModel.load("my_model")
        ```
        """
        with open(path, 'rb') as file:
            concept_model = joblib.load(file)
            return concept_model

    @staticmethod
    def _top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
        """ Return indices of top n values in each row of a sparse matrix
        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix
        Args:
            matrix: The sparse matrix from which to get the top n indices per row
            n: The number of highest values to extract from each row
        Returns:
            indices: The top n indices per row
        """
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
            values = [values[index] if len(values) >= index + 1 else None for index in range(n)]
            indices.append(values)
        return np.array(indices)

    @staticmethod
    def _top_n_values_sparse(matrix: csr_matrix, indices: np.ndarray) -> np.ndarray:
        """ Return the top n values for each row in a sparse matrix
        Args:
            matrix: The sparse matrix from which to get the top n indices per row
            indices: The top n indices per row
        Returns:
            top_values: The top n scores per row
        """
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array([matrix[row, value] if value is not None else 0 for value in values])
            top_values.append(scores)
        return np.array(top_values)