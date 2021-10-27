import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def mmr(cluster_embedding: np.ndarray,
        image_embeddings: np.ndarray,
        indices: List[int],
        top_n: int,
        diversity: float = 0.8) -> List[int]:
    """ Calculate Maximal Marginal Relevance (MMR) between embeddings of
    the candidate images and the cluster embedding.

    MMR considers the similarity of image embeddings with the
    cluster embedding, along with the similarity of already selected
    image embeddings. This results in a selection of images
    that maximize their within diversity with respect to the cluster.

    Arguments:
        cluster_embedding: The cluster embeddings
        image_embeddings: The embeddings of the selected candidate images
        indices: The selected candidate indices
        top_n: The number of images to return
        diversity: How diverse the selected image are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.
    Returns:
         List[int]: The indices of the selected images
    """

    # Extract similarity between images, and between images and their average
    img_cluster_similarity = cosine_similarity(image_embeddings, cluster_embedding)
    image_similarity = cosine_similarity(image_embeddings)

    # Initialize candidates and already choose best images
    images_idx = [np.argmax(img_cluster_similarity)]
    candidates_idx = [i for i in range(len(indices)) if i != images_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and images
        candidate_similarities = img_cluster_similarity[candidates_idx, :]
        target_similarities = np.max(image_similarity[candidates_idx][:, images_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update images & candidates
        images_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [indices[idx] for idx in images_idx]
