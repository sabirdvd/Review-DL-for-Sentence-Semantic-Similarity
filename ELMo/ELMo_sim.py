import tensorflow as tf
import tensorflow_hub as hub

import spacy
#nlp = spacy.load('en_core_web_md')
nlp = spacy.load('en_core_web_md')
import logging
from scipy import spatial

logging.getLogger('tensorflow').disabled = True #OPTIONAL - to disable outputs from Tensorflow

# elmo = hub.Module('path if downloaded/Elmo_dowmloaded', trainable=False)
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

x = ["The president greets the press in Chicago"]
y = ["Obama speaks to the media in Illinois"]

# Extract ELMo features
embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     sess.run(tf.tables_initializer())
     x = sess.run(tf.reduce_mean(embeddings,1))

embeddings1 = elmo(y, signature="default", as_dict=True)["elmo"]

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     sess.run(tf.tables_initializer())
     y = sess.run(tf.reduce_mean(embeddings1,1))


text_similarity = 1 - spatial.distance.cosine(x, y)
print (text_similarity)