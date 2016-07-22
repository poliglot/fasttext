import json
import codecs

ValidationSetSize = 50000

def dataset(training=True):
  i = 0

  with codecs.open('yelp_academic_dataset_review.json', encoding='iso-8859-1') as f:
    if training:
      while i < ValidationSetSize:
        next(f)
        i = i + 1

      for line in f:
        line = line.strip()
        if(len(line.split()) > 50): next(f)
        review = json.loads(line)
        yield (review['text'].encode("utf-8"), int(review['stars']))
    else:
      for line in f:
        line = line.strip()
        if(len(line.split()) > 50): next(f)
        review = json.loads(line)
        yield (review['text'].encode("utf-8"), int(review['stars']))
        i = i + 1
        if i >= ValidationSetSize: break
