import json
import codecs

DataSetPath = 'yelp_academic_dataset_review.json'

def processFile(n, validation):
  with codecs.open(DataSetPath, encoding='iso-8859-1') as f:
    if validation:
      for _ in range(n): next(f)

    for _ in range(n):
      line   = next(f).strip()
      review = json.loads(line)

      while len(review['text'].split()) > 50:
        line   = next(f).strip()
        review = json.loads(line)

      yield (review['text'], int(review['stars']))

def trainingData(n):
  return processFile(n, validation = False)

def validationData(n):
  return processFile(n, validation = True)