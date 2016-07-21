def dataset():
  amazon = open('./sentiment labelled sentences/amazon_cells_labelled.txt', 'r')
  imdb = open('./sentiment labelled sentences/imdb_labelled.txt', 'r')
  yelp = open('./sentiment labelled sentences/yelp_labelled.txt', 'r')

  def grab_data(file, sentences):
    for line in file:
      line = line.strip()
      sentiment = line[-1]
      if sentiment in ['0', '1']:
        sentences.append((sentiment, line[0:-2]))

  sentences = []

  for file in [amazon, imdb, yelp]:
    grab_data(file, sentences)

  return sentences
