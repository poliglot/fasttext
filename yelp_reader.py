import json
import codecs

ValidationSetSize = 400000

def dataset(training=True):
  i = 0

  with codecs.open('./sentiment labelled sentences/yelp_academic_dataset_review.json', encoding='iso-8859-1') as f:
    if training:
      while i < ValidationSetSize:
        next(f)
        i = i + 1

      for line in f:
        line = line.strip()
        review = json.loads(line)
        yield review['text'].encode("utf-8"), review['stars']
    else:
      for line in f:
        line = line.strip()
        review = json.loads(line)
        yield review['text'].encode("utf-8"), review['stars']
        i = i + 1
        if i >= ValidationSetSize: break

# ds = dataset()

# for i in range(5):
#   print(ds.next())
