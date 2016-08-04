# FastText
Unofficial implementation of the paper [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf) by Joulin et al.

## Prerequisites
FastText requires Python 3 with Keras installed.

Obtain the Yelp Dataset from [here](https://www.yelp.com/dataset_challenge) and 
place `yelp_academic_dataset_review.json` in the base directory.

## Training
Train the model using the following command:

```bash
./train.py
```

It generates `data.csv` which represents the model's embedding space of the 
validation set. It is obtained by removing the last layer of the model and using
t-SNE for the dimensionality reduction.

`index.html` implements a D3 visualisation to view the embedding space. You need 
to run a local web server because browsers don't allow file accesses:

```bash
python -m http.server 8000
```

Now point your browser to: [localhost:8000](http://localhost:8080/).

## License
FastText is licensed under the terms of the Apache v2.0 license.

## Authors
* Ihor Kroosh
* Tim Nieradzik
