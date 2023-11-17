import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

#preparing training data for sentence embedding
sym_list = []
with open("symptoms_list.txt", encoding="utf8") as f:
    sym_listUnf = f.read()
    for sym in sym_listUnf.split("\n"):
        if sym.lower() not in sym_list: sym_list.append(sym.lower())

#tagging the data       
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sym_list)]

#implementing the doc2vec network (the following lines just happen to work fine, god do i hope i won't change them by accident :^) )
max_epochs = 100 
vec_size = 20
alpha = 0.025
model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=0) 
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('epoch {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha

model.save("d2v.model")