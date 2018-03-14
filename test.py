import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import re
import datetime

##########
# WORDLIST
##########
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')

wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

# print(len(wordsList))
# print(wordVectors.shape)

# print(wordsList)
# print(wordVectors.shape)

# Search our word list for a word like "baseball", and then access its corresponding vector through the embedding matrix.
# baseballIndex = wordsList.index('baseball')
# wordVectors[baseballIndex]


#################
# SET TEXT STRING
#################
maxSeqLength = 10  # Maximum length of sentence
numDimensions = 300  # Dimensions for each word vector
firstSentence = np.zeros((maxSeqLength), dtype='int32')  # Return a new array of given shape and type, filled with zeros
firstSentence[0] = wordsList.index("i")  # Search vector for word, gets index
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
#firstSentence[8] and firstSentence[9] are going to be 0
print(firstSentence.shape)
print(firstSentence) #Shows the row index for each word

with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)


# GET VECTORS FOR A STRING
def printme( str ):
   # print str
   print(tf.string_split(str))
   return;
# GET VECTORS FOR A STRING


# LOAD TRAINING SETS
# positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
# negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
# numWords = []
# for pf in positiveFiles:
# #     with open(pf, "r", encoding='utf-8') as f:
#     with open(pf, "r") as f:
#         line=f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Positive files finished')
#
# for nf in negativeFiles:
#     with open(nf, "r") as f:
#         line=f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Negative files finished')

# numFiles = len(numWords)
# print('The total number of files is', numFiles)
# print('The total number of words in the files is', sum(numWords))
# print('The average number of words in the files is', sum(numWords)/len(numWords))

# We can also use the Matplot library to visualize this data in a histogram format.
# %matplotlib inline
# plt.hist(numWords, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.axis([0, 1200, 0, 8000])
# plt.show()

# From the histogram as well as the average number of words per file,
# we can safely say that most reviews will fall under 250 words,
# which is the max sequence length value we will set.
maxSeqLength = 250


# Use a single file and transform it into our ids matrix
# fname = positiveFiles[7] #Can use any valid index (not just 3)
# with open(fname) as f:
#     for lines in f:
#         print(lines)
#         exit

# Convert to ids matrix
# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
# strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

firstFile = np.zeros((maxSeqLength), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line=f.readline()  # read line(s)
    cleanedLine = cleanSentences(line)  # remove special chars
    split = cleanedLine.split()  # List of sections in bytes, using sep as the delimiter
    for word in split:
        if indexCounter < maxSeqLength:
            try:
                firstFile[indexCounter] = wordsList.index(word)
            except ValueError:
                firstFile[indexCounter] = 399999 #Vector for unknown words
        indexCounter = indexCounter + 1
firstFile

# Helper functions for training
from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

# Hyperparameters
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

# Import placeholder
# import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

# Get wordvectors
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

# Feed both the LSTM cell and the 3-D tensor full of input data
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

# The first output of the dynamic RNN function can be thought of as the last hidden state vector.
# This vector will be reshaped and then multiplied by a final weight matrix and a bias term to obtain the final output values
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

# define correct prediction and accuracy metrics to track how the network is doing
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# We’ll define a standard cross entropy loss with a softmax layer put on top of the final prediction values.
# For the optimizer, we’ll use Adam and the default learning rate of .001.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# If you’d like to use Tensorboard to visualize the loss and accuracy values, you can also run and the modify the following code.
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

# Load previous training data
# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# saver.restore(sess, tf.train.latest_checkpoint('models'))

# Training
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    # Next Batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    # Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    # Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()

# Load pre-trained model
# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
# saver.restore(sess, tf.train.latest_checkpoint('models'))

# Load movie reviews from test set and get accuracy rating. These are reviews that the model has not been trained on.
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)