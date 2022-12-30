# -*- coding: utf-8 -*-
"""Speaker Identification

"""

import numpy as np
from numpy import loadtxt, array, array, vstack, mean, std
from numpy.linalg import lstsq
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import (Dense, LSTM, Dropout)
from keras_preprocessing.sequence import pad_sequences

class Utils():
    def __init__(self):
       pass

    def read_file(self, block_file, block_label_file=None, file_type='train'):
      if file_type == 'train':
        '''
        Read data from txt file into time series blocks (sequences)
        '''
        block_label_index = loadtxt(block_label_file, delimiter=" ").tolist()
        file = open(block_file, "r")
        speaker_index = 0
        block_index = 0
        block = list()
        blocks = list()
        labels = list()
        for line in file.readlines():
            if line == '\n':
                label = list()
                blocks.append(block)
                label.append(speaker_index)
                labels.append(label)
                block_index += 1
                block = list()
                if speaker_index <= 8 and block_index == block_label_index[speaker_index]:
                    speaker_index += 1
                    block_index = 0
            else:
                point_in_time = list()
                line = line.strip('\n')
                for x in line.split(' ')[:12]:
                    point_in_time.append(float(x))
                block.append(point_in_time)
        return blocks, labels
      else:
        '''
        Read test data from the file and store into time series blocks
        '''
        file = open(block_file, "r")
        speaker_index = 0
        block_index = 0
        block = list()
        blocks = list()
        for line in file.readlines():
            if line == '\n':
                blocks.append(block)
                block_index += 1
                block = list()
            else:
                point_in_time = list()
                line = line.strip('\n')
                for x in line.split(' ')[:12]:
                    point_in_time.append(float(x))
                block.append(point_in_time)
        return blocks

    def pad_to_fixed_size_blocks(self, data_block, max_length, final_block_size):
        '''
        First pad last row till max length, then truncate it to fixed length size
        '''
        # Padding the sequence with the values in last row to max length
        fixed_size_block = []
        for block in data_block:
            block_len = len(block)
            last_row = block[-1]
            n = max_length - block_len

            to_pad = np.repeat(block[-1], n).reshape(12, n).transpose()
            new_block = np.concatenate([block, to_pad])
            fixed_size_block.append(new_block)

        final_dataset = np.stack(fixed_size_block)

        # truncate the sequence to final_block_size
        final_dataset = pad_sequences(final_dataset, maxlen=final_block_size, padding='post', dtype='float', truncating='post')

        return final_dataset

    def convert_to_vectors(self, data_block, block_label, final_block_size):
        '''
        Convert fixed size block to feature vectors for ML algorithms
        '''
        block_label = [i[0] for i in block_label]
        # print(block_label)
        vectors = list()
        n_features = 12
        for i in range(len(data_block)):
            block = data_block[i]
            vector = list()
            for row in range(1, final_block_size+1):
                for col in range(n_features):
                    vector.append(block[-row, col])

            vector.append(block_label[i])
            vectors.append(vector)
        vectors = array(vectors)
        vectors =vectors.astype('float32')
        return vectors
    
    def balance_classes(self, train_data):
      '''
      Balance the train data
      '''
      speaker_0 = train_data[train_data[:,-1] == 0]
      speaker_4 = train_data[train_data[:,-1] == 4]
      speaker_1 = train_data[train_data[:,-1] == 1]
      speaker_5 = train_data[train_data[:,-1] == 5]
      speaker_8 = train_data[train_data[:,-1] == 8]
      others = train_data[np.logical_or.reduce((train_data[:,-1] == 2, train_data[:,-1] == 3, train_data[:,-1] == 6, train_data[:,-1] == 7))]

      upsample_0 = resample(speaker_0, replace=True, n_samples=40, random_state=123)
      upsample_1 = resample(speaker_1, replace=True, n_samples=40, random_state=123)
      upsample_4 = resample(speaker_4, replace=True, n_samples=40, random_state=123)
      upsample_5 = resample(speaker_5, replace=True, n_samples=40, random_state=123)
      upsample_8 = resample(speaker_8, replace=True, n_samples=40, random_state=123)

      args = (others, upsample_0, upsample_1, upsample_4, upsample_5, upsample_8)
      balanced_train_data = np.concatenate(args)

      return balanced_train_data

class DisplayUtils():
  def display_lpc_distribution(self, train_blocks):
      '''
      Visualize all 12 lpc coefficients distribution over all blocks
      '''
      point_in_time = vstack(train_blocks)
      plt.figure(figsize=(10, 25))
      plt.title('LPC coefficients  Distribution')
      coefficients = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
      for c in coefficients:
          plt.subplot(len(coefficients), 1, c + 1)
          plt.hist(point_in_time[:, c], bins=100)
      plt.savefig('lpc_coeff_dist.png')
      plt.show()

  def display_block_length_distribution(self, train_blocks):
      '''
      visualize the distribution of block length
      '''
      points_in_time = [len(x) for x in train_blocks]
      plt.title('Block Length Distribution')
      plt.hist(points_in_time, bins=25)
      plt.savefig('block_len_dist.png')
      plt.show()

  def lpc_scatter_plot(self, final_train_data):
      '''
      use scatter plot to visualize the grouping of users
      '''
      train_X, train_y = final_train_data[:, :-1], final_train_data[:, -1]
      colormap = get_cmap('viridis')
      colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1 / (6 - 1))]
      fig = plt.figure()
      ax = fig.gca(projection='3d')
      ax = Axes3D(fig)
      ax.scatter(train_X[:, 0], train_X[:, 1], train_X[:, 2], c=train_y, s=50, cmap=mat.colors.ListedColormap(colors))
      plt.title('Speaker Plot')
      plt.savefig('scatter_plot.png')
      plt.show()

  def display_lpc_time_series(self, speaker_blocks):
      '''
      visualize a block of lpc series for each speaker
      '''
      # group sequences by speaker
      speakers = [i + 1 for i in range(0,9)]
      speakers_voice = {}
      for speaker in speakers:
          speakers_voice[speaker] = [speaker_blocks[j] for j in range(len(speakers)) if speakers[j] == speaker]
      plt.figure(figsize=(10, 35))
      plt.title('LPC trend for each speaker')
      for i in speakers:
          plt.subplot(len(speakers), 1, i)
          coeff_series = vstack(speakers_voice[i][0])
          for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
              plt.plot(coeff_series[:, j], label='test')
          plt.title('Speaker ' + str(i), y=0, loc='left')
      plt.savefig('lpc_series.png')
      plt.show()

  def regress(self, y):
      X = array([i for i in range(len(y))]).reshape(len(y), 1)
      b = lstsq(X, y)[0][0]
      yhat = b * X[:, 0]
      return yhat

  def display_fitted_lpc_series(self, speaker_blocks):
      '''
      visualize a fitted lpc series for each speaker to see the trend
      '''
      speakers = [i + 1 for i in range(0,9)]
      speakers_voice = {}
      for speaker in speakers:
          speakers_voice[speaker] = [speaker_blocks[j] for j in range(len(speakers)) if speakers[j] == speaker]
      plt.figure(figsize=(10, 25))
      plt.title('LPC trend for each speaker')
      for i in speakers:
          plt.subplot(len(speakers), 1, i)
          coeff_series = vstack(speakers_voice[i][0])
          plt.plot(coeff_series[:, i])
          plt.plot(self.regress(coeff_series[:, i]))
          plt.title('Speaker ' + str(i), y=0, loc='left')
      plt.savefig('fitted_lpc_series.png')
      plt.show()

  def show_training_history(self, history):
      '''
      Show LSTM model training loss over each epoch
      '''
      plt.figure(figsize=(8, 6))
      plt.plot(history.history['loss'])
      plt.title('LSTM model training loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.savefig('lstm_train_error.png')
      plt.show()
  
  def display_class_distribution(self, train_block_label, disp_type="orginal"):
    '''
    Show the class distribution
    '''
    unique, counts = np.unique(train_block_label, return_counts=True)
    label_counts = dict(zip(unique, counts))
    plt.figure(figsize=(10,6))
    sns.barplot(x = unique, y = counts)
    plt.title('Class distribution ('+disp_type+')')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig(disp_type+'class_distribution.png')
    plt.show()

class Models():
    def __init__(self):
        pass

    # Function to print result of the classifier
    def print_result(self, y_test, y_pred):
        '''
        print classification result (on train set)
        '''
        print("\n\n")
        print(metrics.classification_report(y_test, y_pred))
        print("Confusion Matrix: \n\n", metrics.confusion_matrix(y_test, y_pred))

    def save_prediction(self, test_X, predict, file_name):
        '''
        save prediction to the csv file
        '''
        predictions = []
        for block in range(0, len(test_X)):
            predictions.append([block,int(predict[block])])
        submission = pd.DataFrame(predictions, columns=['block_num', 'prediction'])
        submission.to_csv(file_name, header=['block_num','prediction'], index=None, sep=',')

    def build_train_lstm_model(self, trainX, trainy):
        '''
        build lstm model and train
        '''
        trainy = to_categorical(trainy)
        verbose, epochs, batch_size = 2, 150, 64
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()
        model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
        model.save('lstm_model.h5')
        return history

    def lstm_predict(self, testX):
        '''
        load the saved model and predict the class for test data
        '''
        model = load_model('lstm_model.h5')
        predict_classes = []
        for pred in model.predict(testX):
          predict_classes.append(np.argmax(pred))

        return np.array(predict_classes)

    def compare_ml_models(self, train_X, train_y, nfold):
        '''
        Initial experimentation on several algorithms
        '''
        models, names = list(), list()
        # knn
        models.append(KNeighborsClassifier())
        names.append('KNN')
        # logistic
        models.append(LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=200))
        names.append('LR')
        # cart
        models.append(DecisionTreeClassifier())
        names.append('CART')
        # svm
        models.append(SVC())
        names.append('SVM')
        # random forest
        models.append(RandomForestClassifier(n_estimators=100))
        names.append('RF')
        # evaluate models

        all_scores = list()
        for i in range(len(models)):
            s = StandardScaler()
            p = Pipeline(steps=[('s', s), ('m', models[i])])
            scores = cross_val_score(p, train_X, train_y, scoring='accuracy', cv=nfold, n_jobs=-1)
            all_scores.append(scores)
            m, s = mean(scores) * 100, std(scores) * 100
            print('%s %.3f%% +/-%.3f' % (names[i], m, s))

        return all_scores, names

    def svm_parameter_tuning(self, train_X, train_y, nfold):
        '''
        find best parameter for svm
        '''
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel':['rbf','linear']}
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfold, verbose=1)
        grid_search.fit(train_X, train_y)
        grid_search.best_params_

        scores = cross_val_score(grid_search.best_estimator_, train_X, train_y, scoring='accuracy', cv=nfold, n_jobs=-1)
        m, s = mean(scores) * 100, std(scores) * 100
        print('%s %.3f%% +/-%.3f' % ('SVM fine tuned', m, s))


        return grid_search.best_params_, grid_search.best_estimator_


  

    def test_best_model(self, best_estimator, train_X, train_y):
        '''
        Test the best model
        '''
        y_pred = cross_val_predict(best_estimator, train_X, train_y, cv=5)
        print("Performance by SVM (with best estimator) in test data using cross-val...\n\n")
        self.print_result(train_y, y_pred)

    def classify_test_data(self, best_estimator, train_X, train_y, test_X):
        '''
        Classify test data using best svm settings
        '''
        best_estimator.fit(train_X, train_y)
        predict = best_estimator.predict(test_X)
        return predict

    def run_classification_models(self, train_data, test_data):
        '''
        use traditional machine learning approach
        '''
        train_X, train_y = train_data[:,:-1], train_data[:,-1]
        test_X, test_y = test_data[:, :-1], test_data[:, -1]
        nfold = 5

        print("Running Algorithms for Spot Checking ... \n\n")
        all_scores, names = self.compare_ml_models(train_X, train_y, nfold)

        # # Visualize boxplot to see the best model
        plt.boxplot(all_scores, labels=names)
        # pyplot.show()
        plt.savefig('spot_check_box_plot.png')

        # Since SVM shows the best performance.. Let's tune the parameter for SVM and find the best model
        print("Running Grid Search for the Best Algorithm (SVM) ... \n\n")
        best_param, best_estimator = self.svm_parameter_tuning(train_X, train_y, nfold)
        print("Best Parameters:      ", best_param)
        print("\n\nBest Estimators:      ", best_estimator)

        # test the performance of best svm model
        self.test_best_model(best_estimator, train_X, train_y)
        print("Predicting Speakers.....\n\n")
        #predict the speaker for test data using best svm model
        predict = self.classify_test_data(best_estimator, train_X, train_y, test_X)
        self.save_prediction(test_X, predict, 'submission.csv')

    def run_LSTM_model(self, trainX, trainy, testX):
        '''
        use lstm based classifer for speaker classification
        '''
        # build and train LSTM Model
        print("\n\nTraining LSTM....\n\n")
        history = self.build_train_lstm_model(trainX, trainy)

        # visualize the training error and convergence of LSTM
        display = DisplayUtils()
        display.show_training_history(history)

        #predict the speaker using
        predict = self.lstm_predict(testX)
        self.save_prediction(testX, predict, 'submission_lstm.csv')

utils = Utils()

train_data = 'train.txt'
test_data = 'test.txt'
train_label = 'train_block_labels.txt'

# load data
print("Loading Data\n\n")
train_block, train_block_label = utils.read_file(block_file = train_data, block_label_file = train_label, file_type='train')
test_block = utils.read_file(block_file = test_data, file_type='test')

# explore data, do some visualization
print("Exploring Data\n\n")
display = DisplayUtils()

# histogram for the lpc coefficient distribution
display.display_lpc_distribution(train_block)

# histogram for the block length (or point of time) distribution
display.display_block_length_distribution(train_block)

# plot one block of lpc coefficient for each speaker to look at the pattern of voice frequency
display.display_lpc_time_series(train_block)
display.display_fitted_lpc_series(train_block)

max_length = 29
final_block_size = 18

print("Data Preprocessing (padding to fixed size blocks)\n\n")
# Take the best lengths (18), truncate the longer block, and pad the  shorter block by the last row
train_data = utils.pad_to_fixed_size_blocks(train_block, max_length, final_block_size)
test_data = utils.pad_to_fixed_size_blocks(test_block, max_length, final_block_size)

# dummy test label for convenience
test_block_label = [[i] for i in np.zeros(len(test_data))]

print("Generating Features (for ML Algorithms)\n\n")

# Generate fixed length feature vector for traditional machine learning input
final_train_data = utils.convert_to_vectors(train_data, train_block_label, final_block_size)
final_test_data = utils.convert_to_vectors(test_data, test_block_label, final_block_size)


# Class distribution before balancing
display.display_class_distribution(final_train_data[:,-1], disp_type='orginal')


print("Before balancing")
model = Models()
model.run_classification_models(final_train_data, final_test_data)

# Balancing the classes for training
final_train_data = utils.balance_classes(final_train_data)

# Class distribution after balancing
display.display_class_distribution(final_train_data[:,-1], disp_type='balanced')

# Scatter plot to figure if there is any grouping based on feature vector
display.lpc_scatter_plot(final_train_data)

# It appears that there is clustering, so classifying with few algorithms
model = Models()
model.run_classification_models(final_train_data, final_test_data)

print("SVM Prediction Saved (see 'submission.csv' )\n\n")

# Also try LSTM for classification
model.run_LSTM_model(np.array(train_data), np.array(train_block_label), np.array(test_data))
print("LSTM Prediction Saved (see 'submission_lstm.csv' )\n\n")