import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import re
import datetime
from random import randint
import getopt, sys
# import logging
from tqdm import tqdm
import os
from tensorflow.python.lib.io import file_io

###############
# CLI ARGUMENTS
###############
fullCmdArguments = sys.argv  # read commandline arguments, first
argumentList = fullCmdArguments[1:]  # - further arguments
# print(argumentList)

# Prepare valid parameters
unixOptions = "ho:v"
gnuOptions = ["help", "test", "train", "accuracy", "load", "savemodel"]

# Parse the argument list
try:
    arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))
    sys.exit(2)

# Evaluate variables
for currentArgument, currentValue in arguments:
    if currentArgument in ("-h", "--help"):
        print ("Displaying help")
    elif currentArgument in ("-t", "--test"):
        print ("Running test")
        testSet = 1
    elif currentArgument in ("-T", "--train"):
        print ("Running training")
        trainSet = 1
    elif currentArgument in ("-a", "--accuracy"):
        print ("Analyzing accuracy...")
        accuracySet = 1
    elif currentArgument in ("-l", "--load"):
        print ("Loading pre-trained data...")
        loadpreSet = 1
    elif currentArgument in ("-s", "--savemodel"):
        print ("Preparing to save model...")
        saveModelSet = 1

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


def cleanSentences(string):
    print ('\nCleaning sentence (remove special chars)...')
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


# GET VECTORS FOR A STRING (MJC TEST)
def printme(teststring):
    print('\n\n-- GET VECTORS FOR A STRING: ' + teststring)

    # 1a. Split sentence string to a list
    words = teststring.split()
    print('words: ')
    print(words)

    # Try3: Build array via loop
    # testArray = np.array([])
    testArray = []
    testArrayToNumpy = np.array([])
    i = 0
    for loop in words:
        i = i + 1
        print('loop: ')
        print(loop)

        # Search vocab and convert to integer
        convertedWord = wordsList.index(loop)
        # convertedWord2 = np.rint(convertedWord)
        print('\n^- convertedWord:')
        print(convertedWord)

        # testArray = np.append(testArray, convertedWord)  # https://stackoverflow.com/a/28944075 (BROKEN)
        # testArray = np.append(testArray, np.rint(convertedWord))  # https://stackoverflow.com/a/28944075 (BROKEN)
        testArray.insert(i, convertedWord)  # Python array (LAST WORKING)
        testArrayToNumpy = np.array(testArray)

    print('\n\n*- testArray: ')
    
    print(testArrayToNumpy)
    return testArrayToNumpy


def _write_assets(assets_directory, assets_filename):
  """Writes asset files to be used with SavedModel for half plus two.
  Args:
    assets_directory: The directory to which the assets should be written.
    assets_filename: Name of the file to which the asset contents should be
        written.
  Returns:
    The path to which the assets file was written.
  """
  if not file_io.file_exists(assets_directory):
    file_io.recursive_create_dir(assets_directory)

  path = os.path.join(
      tf.compat.as_bytes(assets_directory), tf.compat.as_bytes(assets_filename))
  file_io.write_string_to_file(path, "asset-file-contents")
  return path


###############
# LOAD WORDLIST
###############
wordsList = np.load('wordsList.npy')
print('Loaded word list!')

wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('^- Loaded word vectors!')

# print(len(wordsList))
# print(wordVectors.shape)

# print(wordsList)
# print(wordVectors.shape)

# Search our word list for a word like "baseball", then access
# its corresponding vector through the embedding matrix.
# baseballIndex = wordsList.index('baseball')
# wordVectors[baseballIndex]


#######################
# TEST: SET TEXT STRING
#######################
maxSeqLength = 10  # Maximum length of sentence
numDimensions = 300  # Dimensions for each word vector (ORIGINAL)
# numDimensions = 50  # Dimensions for each word vector
if 'testSet' in locals():
    print ('TESTING: NOW RUNNING...')
    # maxSeqLength = 10  # Maximum length of sentence
    # numDimensions = 300  # Dimensions for each word vector

    # Return a new array of given shape and type, filled with zeros
    firstSentence = np.zeros((maxSeqLength), dtype='int32')

    firstSentence[0] = wordsList.index("i")  # Search vector for word, gets index
    firstSentence[1] = wordsList.index("thought")
    firstSentence[2] = wordsList.index("the")
    firstSentence[3] = wordsList.index("movie")
    firstSentence[4] = wordsList.index("was")
    firstSentence[5] = wordsList.index("incredible")
    firstSentence[6] = wordsList.index("and")
    firstSentence[7] = wordsList.index("inspiring")
    #firstSentence[8] and firstSentence[9] are going to be 0
    print('\nfirstSentence: ')
    print(firstSentence)

    firstSentence = printme("i thought the movie was incredible and inspiring")
    print('\nfirstSentence: ')
    print(firstSentence)

    print('\n\n firstSentence as words array: ')
    print(firstSentence)  # Shows words from sentence string as array

    print('\n\n firstSentence as vector ids array: ')
    print(firstSentence.shape)  # Returns the shape of a tensor
    print(firstSentence)  # Shows the row index for each word

    with tf.Session() as sess:
        # print ('^- TEST: CONVERTED STRING:')
        # Looks up ids in a list of embedding tensors
        print(tf.nn.embedding_lookup(wordVectors, firstSentence).eval().shape)


##############
# LOAD REVIEWS
##############
print('\n\nLOAD REVIEW (POSITIVE & NEGATIVE) FILES...')
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    # with open(pf, "r", encoding='utf-8') as f:
    with open(pf, "r") as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('^- Positive files finished')

for nf in negativeFiles:
    with open(nf, "r") as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('^- Negative files finished')

numFiles = len(numWords)
print('^- The total number of files is', numFiles)
print('^- The total number of words in the files is', sum(numWords))
print('^- The average number of words in the files is', sum(numWords)/len(numWords))


#######################################################
# Visualize data as histogram using the Matplot library
#######################################################
# print ('Visualize data as histogram using the Matplot library...')
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


########################################################
# Use a single file and transform it into our ids matrix
########################################################
print ('\n\nCONVERTING SINGLE FILE TO ID MATRIX...')
fname = positiveFiles[7]  # Can use any valid index (not just 3)

with open(fname) as f:
    for lines in f:
        print(lines)
        exit


#######################
# Convert to ids matrix
# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
#######################
firstFile = np.zeros((maxSeqLength), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line=f.readline()  # read line(s)
    cleanedLine = cleanSentences(line)  # remove special chars
    split = cleanedLine.split()  # List of sections in bytes, using sep as the delimiter
    for word in split:
        # print('for word...')
        if indexCounter < maxSeqLength:
            try:
                firstFile[indexCounter] = wordsList.index(word)
            except ValueError:
                firstFile[indexCounter] = 399999  # Vector for unknown words
        indexCounter = indexCounter + 1
firstFile


##########################################
# CONVERT ALL 50K STRINGS TO AN IDS MATRIX
##########################################
# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
#    with open(pf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1

# for nf in negativeFiles:
#    with open(nf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
# #Pass into embedding function and see if it evaluates.
# np.save('idsMatrix', ids)
ids = np.load('idsMatrix.npy')


# Helper functions for training
def getTrainBatch():
    # print('getTrainBatch...')
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
    # print('getTestBatch...')
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


def export_saved_model(version, path, sess=None):
    tf.app.flags.DEFINE_integer('version', version, 'version number of the model.')
    tf.app.flags.DEFINE_string('work_dir', path, 'your older model  directory.')
    # tf.app.flags.DEFINE_string('model_dir', '/tmp/model_name', 'saved model directory')
    tf.app.flags.DEFINE_string('model_dir', 'tmp/model_name', 'saved model directory')
    FLAGS = tf.app.flags.FLAGS

    # you can give the session and export your model immediately after training
    if not sess:
        saver = tf.train.import_meta_graph(os.path.join(path, 'pretrained_lstm.ckpt-2000.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(path))

    export_path = os.path.join(
        tf.compat.as_bytes(FLAGS.model_dir),
        tf.compat.as_bytes(str(FLAGS.version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # define the signature def map here
    feature_configs = {
        'x': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'y': tf.FixedLenFeature(shape=[], dtype=tf.string)
        # 'x': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        # 'y': tf.FixedLenFeature(shape=[], dtype=tf.float32)
    }
    serialized_example = tf.placeholder(tf.string, name="tf_example")
    # serialized_example = tf.placeholder(tf.int32, name="tf_example") # no good
    # serialized_example = tf.placeholder(tf.int32, name="input_data")
    tf_example = tf.parse_example(serialized_example, feature_configs)
    x = tf.identity(tf_example['x'], name='x')
    y = tf.identity(tf_example['y'], name='y')
    predict_input = x
    predict_output = y
    predict_signature_def_map = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs={
            tf.saved_model.signature_constants.PREDICT_INPUTS: predict_input
        },
        outputs={
            tf.saved_model.signature_constants.PREDICT_OUTPUTS: predict_output
        }
    )

    # SavedModel.
    # # Assets: Create an assets file that can be saved and restored as part of the
    # original_assets_directory = "/home/trzn/Documents/AIWork/LSTM-Sentiment-Analysis/models"
    # # original_assets_directory = path
    # original_assets_filename = "foo.txt"
    # original_assets_filepath = _write_assets(original_assets_directory,
    #                                          original_assets_filename)
    #
    # # Assets: Set up the assets collection.
    # assets_filepath = tf.constant(original_assets_filepath)
    # tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, assets_filepath)
    # filename_tensor = tf.Variable(
    #     original_assets_filename,
    #     name="filename_tensor",
    #     trainable=False,
    #     collections=[])
    # assign_filename_op = filename_tensor.assign(original_assets_filename)

    # define the signature def map here
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        # signature_def_map={
        #     # 'predict_xxx': predict_signature_def_map
        #     'predict_text': predict_signature_def_map
        # },
        signature_def_map=predict_signature_def_map,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        legacy_init_op=legacy_init_op  # ORIG
        # legacy_init_op=tf.group(assign_filename_op)
    )

    builder.save()
    print('Export SavedModel!')


# Hyperparameters
batchSize = 24
lstmUnits = 64
numClasses = 2
# iterations = 100000 (ORIGINAL)
iterations = 50000

# Import placeholder
print('\n\nINITIALIZING..')
print('^- Import placeholder...')
tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

# Get wordvectors
print('^- Fetching wordvectors...')
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

# Feed both the LSTM cell and the 3D tensor full of input data
print('^- Feeding LSTM cell & 3D tensor w/ input data...')
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

# Define correct prediction and accuracy metrics to track how the network is doing
print('^- Define correct prediction and accuracy metrics to track how the network is doing...')
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# We'll define a standard cross entropy loss with a softmax layer put on top of the final prediction values.
# For the optimizer, we'll use Adam and the default learning rate of .001.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


#####################################################################
# If you'd like to use Tensorboard to visualize the loss and accuracy
# values, you can also run and the modify the following code.
#####################################################################
with tf.Session() as sess:  # New
    # print('\n\nEnabling Tensorboard support...')
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)


########################
# Load pre-trained model
########################
if 'loadpreSet' in locals():
    print('\n\nLOADING PRE-TRAINED MODEL...')
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('models'))


# Training
if 'trainSet' in locals():
    print('\n\ntrainSet...')
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # for i in range(iterations):
    for i in tqdm(range(iterations)):
        # print('^- Running batch...')  # Spammy
        # Next Batch of reviews
        nextBatch, nextBatchLabels = getTrainBatch();
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        # Write summary to Tensorboard
        if (i % 50 == 0):
            # print('^- Writing summary to Tensorboard...')  # Spammy
            summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
            writer.add_summary(summary, i)
            # logging.getLogger().setLevel(logging.INFO)

        # Save the network every 10,000 training iterations
        # if (i % 10000 == 0 and i != 0):
        if (i % 1000 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
            # Also save 'savedModel'
            export_saved_model(1, 'models/', sess)
    writer.close()


# Load movie reviews from test set and get accuracy rating. These are reviews that the model has not been trained on.
# if FLAGS.accuracy:
if 'accuracySet' in locals():
    print('\n\nRUNNING ACCURACY TESTS...')
    iterations = 10

    for i in range(iterations):
    # for i in tqdm(range(iterations)):
        nextBatch, nextBatchLabels = getTestBatch()
        print("^- Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)


#
if 'saveModelSet' in locals():
    print('\n\nSAVING MODEL...')
    # export_dir = "Saved_Models/"
    # # ...
    # builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    # with tf.Session(graph=tf.Graph()) as sess:
    #     # ...
    #     builder.add_meta_graph_and_variables(sess,
    #                                          [tag_constants.TRAINING],
    #                                          signature_def_map=foo_signatures,
    #                                          assets_collection=foo_assets)
    # # ...
    # # Add a second MetaGraphDef for inference.
    # with tf.Session(graph=tf.Graph()) as sess:
    #     # ...
    #     builder.add_meta_graph([tag_constants.SERVING])
    # # ...
    # builder.save()
    # export_saved_model(1, '/home/trzn/Documents/AIWork/LSTM-Sentiment-Analysis/models/')
    export_saved_model(1, '/home/trzn/Documents/AIWork/LSTM-Sentiment-Analysis/models/', sess)
