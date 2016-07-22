import json
import codecs

# ValidationSetSize = 400000
ValidationSetSize = 1000

def dataset(training=True):
  i = 0

  # with codecs.open('./sentiment labelled sentences/yelp_academic_dataset_review.json', encoding='iso-8859-1') as f:
  with codecs.open('./sentiment labelled sentences/test.json', encoding='iso-8859-1') as f:
    if training:
      while i < ValidationSetSize:
        next(f)
        i = i + 1

      for line in f:
        line = line.strip()
        review = json.loads(line)
        yield (review['text'].encode("utf-8"), int(review['stars']))
    else:
      for line in f:
        line = line.strip()
        review = json.loads(line)
        yield (review['text'].encode("utf-8"), int(review['stars']))
        i = i + 1
        if i >= ValidationSetSize: break

# ds = dataset()

# for i in range(5):
#   print(ds.next())
