import json

def dataset():
  file = open('./sentiment labelled sentences/yelp_academic_dataset_review.json', 'r')

  for line in file:
    line = line.strip()
    review = json.loads(line)
    yield (review['text'], review['stars'])

# ds = dataset()

# for i in range(5):
#   print(ds.next())
