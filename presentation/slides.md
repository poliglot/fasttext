---
title: FastText
author: Ihor Kroosh, Tim Nieradzik
date: 22th July 2016
institute: Ukrainian Catholic University
lang: british
spelling: new
include-after:
    - \plain{Questions?}
---

## Goal
\centering
\huge

Improve speed of training models for Sentiment Analysis

##
\centering
\huge

Bag of Tricks for Efficient Text Classification

##
\centering
\huge

Hashing of n-grams

## Hashing
**2-grams:** $(S_{t - 1} \cdot P_1)\ \text{mod}\ N$

**3-grams:** $(S_{t - 2} \cdot P_1 \cdot P_2 + S_{t - 1} \cdot P_1)\ \text{mod}\ N$

$t$: Current word

$S$: Word indices

$N$: Number of buckets in hashing vector

$P_n$: Large random prime number

## Code
\centering
\large

**Available on GitHub**

\huge

[github.com/poliglot/fasttext](https://github.com/poliglot/fasttext)
