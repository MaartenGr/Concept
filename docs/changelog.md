## **Version 0.2.1**
*Release date:  5 November, 2021*

* Fixed issue when loading in more than 40.000 images
* Fixed `transform` only working with pre-trained embeddings


## **Version 0.2.0**
*Release date:  2 November, 2021*

Added **c-TF-IDF** as an algorithm to extract textual representations from images.

```python
from concept import ConceptModel

concept_model = ConceptModel(ctfidf=True)
concepts = concept_model.fit_transform(img_names, docs=docs)
```

From the textual and visual embeddings, we use cosine similarity to find the best matching words 
for each image. Then, after clustering the images, we combine all words in a cluster into a single 
documents. Finally, c-TF-IDF is used to find the best words for each concept cluster. 

The benefit of this method is that it takes the entire cluster structure into account when creating the 
representations. This is not the case when we only consider words close to the concept embedding.

## **Version 0.1.1**
*Release date:  31 October, 2021*

* Fix RAM issues
* Update documentation
* Add `ftfy` dependency
* Fix `.visualize_concepts`
* Added `.search_concepts`

## **Version 0.1.0**
*Release date:  27 October, 2021*

* Update Readme with small example
* Create documentation page: https://maartengr.github.io/Concept/
* Fix `fit` not working properly
* Better visualization of resulting concepts

## **Version 0.0.1**
*Release date:  27 October, 2021*

The first release of Concept Modeling 🥳, a technique that allows for topic modeling of 
images and text together.