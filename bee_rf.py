from sklearn import tree, metrics
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from skimage.feature import hog
from skimage import exposure
import glob
import cv2
import os
import pickle

# For Ignoring any divide by zero errors.
np.seterr(divide='ignore', invalid='ignore')

def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

# Fit function for testing an image with the saved random forest classifiers.
# returns [0] for BEE and [1] for NO-BEE
def fit_image_rf(rf,image_path):
    img = cv2.imread(image_path)
    img2 = cv2.resize(img,(90,90))
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    features,hog_image = hog(gray, 
                        orientations=9, 
                        pixels_per_cell=(4,4), 
                        cells_per_block=(1,1), 
                        transform_sqrt=False,
                        visualize=True,
                        feature_vector=False)                            
    hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,10))
    min = np.min(hog_image_rescaled)
    max = np.max(hog_image_rescaled)
    hog_image_rescaled = (hog_image_rescaled - min)/(max-min)
    scaled_image = np.array(hog_image_rescaled)
    scaled_image = scaled_image.reshape((8100,))
    scaled_image = scaled_image.reshape(1,-1)
    prediction = rf.predict(scaled_image)
    return prediction

# rf = load('rf_BEE2_1S_'+str(5))
# rf = load('rf_BEE2_2S_'+str(num_trees))
# image_path = '/home/nikhilganta/Documents/RF_FE/BEE2_1S/one_super/training/bee/1/56970.png'
# print(fit_image_rf(rf,image_path))

def bee_image_processing(dataType):
    bee21Spath = '/home/nikhilganta/Documents/RF_FE/BEE2_1S'
    bees = glob.glob(bee21Spath + '/*/'+dataType+'/bee/*/*.png')
    nonbees = glob.glob(bee21Spath + '/*/'+dataType+'/no_bee/*/*.png')

    for file in bees:
        img = cv2.imread(file)
        img2 = cv2.resize(img,(90,90))
        gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        features,hog_image = hog(gray, 
                            orientations=9, 
                            pixels_per_cell=(4,4), 
                            cells_per_block=(1,1), 
                            transform_sqrt=False,
                            visualize=True,
                            feature_vector=False)                            
        hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,10))
        min = np.min(hog_image_rescaled)
        max = np.max(hog_image_rescaled)
        hog_image_rescaled = (hog_image_rescaled - min)/(max-min)
        hog_image_rescaled = hog_image_rescaled*255
        hog_image_rescaled = hog_image_rescaled.astype(np.uint8)
        save_path = '/home/nikhilganta/Documents/RF_FE/BEE2_1S_HOG/' + dataType + '/BEE/Images/'
        cv2.imwrite(save_path + os.path.basename(file),hog_image_rescaled)
    for file in nonbees:
        img = cv2.imread(file)
        img2 = cv2.resize(img,(90,90))
        gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        features,hog_image = hog(gray, 
                            orientations=9, 
                            pixels_per_cell=(4,4), 
                            cells_per_block=(1,1), 
                            transform_sqrt=False,
                            visualize=True,
                            feature_vector=False)                            
        hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,10))
        min = np.min(hog_image_rescaled)
        max = np.max(hog_image_rescaled)
        hog_image_rescaled = (hog_image_rescaled - min)/(max-min)
        hog_image_rescaled = hog_image_rescaled*255
        hog_image_rescaled = hog_image_rescaled.astype(np.uint8)
        save_path = '/home/nikhilganta/Documents/RF_FE/BEE2_1S_HOG/' + dataType + '/NO-BEE/Images/'
        cv2.imwrite(save_path + os.path.basename(file),hog_image_rescaled)

    bee22Spath = '/home/nikhilganta/Documents/RF_FE/BEE2_2S'
    bees = glob.glob(bee22Spath + '/*/'+dataType+'/bee/*/*.png')
    nonbees = glob.glob(bee22Spath + '/*/'+dataType+'/no_bee/*/*.png')

    for file in bees:
        img = cv2.imread(file)
        img2 = cv2.resize(img,(90,90))
        gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        features,hog_image = hog(gray, 
                            orientations=9, 
                            pixels_per_cell=(4,4), 
                            cells_per_block=(1,1), 
                            transform_sqrt=False,
                            visualize=True,
                            feature_vector=False)                            
        hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,10))
        min = np.min(hog_image_rescaled)
        max = np.max(hog_image_rescaled)
        hog_image_rescaled = (hog_image_rescaled - min)/(max-min)
        hog_image_rescaled = hog_image_rescaled*255
        hog_image_rescaled = hog_image_rescaled.astype(np.uint8)
        save_path = '/home/nikhilganta/Documents/RF_FE/BEE2_2S_HOG/' + dataType + '/BEE/Images/'
        cv2.imwrite(save_path + os.path.basename(file),hog_image_rescaled)
    for file in nonbees:
        img = cv2.imread(file)
        img2 = cv2.resize(img,(90,90))
        gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        features,hog_image = hog(gray, 
                            orientations=9, 
                            pixels_per_cell=(4,4), 
                            cells_per_block=(1,1), 
                            transform_sqrt=False,
                            visualize=True,
                            feature_vector=False)                            
        hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,10))
        min = np.min(hog_image_rescaled)
        max = np.max(hog_image_rescaled)
        hog_image_rescaled = (hog_image_rescaled - min)/(max-min)
        hog_image_rescaled = hog_image_rescaled*255
        hog_image_rescaled = hog_image_rescaled.astype(np.uint8)
        save_path = '/home/nikhilganta/Documents/RF_FE/BEE2_2S_HOG/' + dataType + '/NO-BEE/Images/'
        cv2.imwrite(save_path + os.path.basename(file),hog_image_rescaled)

# For processing the images and saving the images in separate folder.

# bee_image_processing('training')
# print('training images created')
# bee_image_processing('testing')
# print('testing images created')
# bee_image_processing('validation')
# print('validation images created')


# bee_process function loads all the bees and no-bees using the glob library and the data labelling is done.

def bee_process(dataType):
    dataX = []
    dataY = []

    # Uncomment this for the BEE2_1S Dataset

    # bee21Spath = '/home/nikhilganta/Documents/RF_FE/BEE2_1S_HOG/'
    # bees = glob.glob(bee21Spath + dataType + '/BEE/Images/*.png')
    # nonbees = glob.glob(bee21Spath + dataType + '/NO-BEE/Images/*.png')

    # Uncomment this for the BEE2_2S Dataset

    # bee22Spath = '/home/nikhilganta/Documents/RF_FE/BEE2_2S_HOG/'
    # bees = glob.glob(bee22Spath + dataType + '/BEE/Images/*.png')
    # nonbees = glob.glob(bee22Spath + dataType + '/NO-BEE/Images/*.png')

    filePaths = []
    for file in bees:
        filePaths.append((file,0))
    for file in nonbees:
        filePaths.append((file,1))
    filePaths = shuffle(filePaths)
    for i in range(len(filePaths)):
        image = cv2.imread(filePaths[i][0])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        scaled_image = image/255.0
        scaled_image = np.array(scaled_image)
        dataX.append(scaled_image)
        dataY.append(filePaths[i][1])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX,dataY

# This  load_data() function calls the bee_process() function which prepares the training, testing and validation data individually.

def load_data():
    training_data = bee_process('training')
    validation_data = bee_process('validation')
    testing_data = bee_process('testing')
    return training_data, validation_data, testing_data

# This  load_data_wrapper() function calls the load_data() function and transforms the data into the required form individually.

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (8100, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (8100, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (8100, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 2-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    bee and no-bee into a corresponding desired output"""
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

# train_data, valid_data, test_data = load_data_wrapper()

# Initializing the zero matrices for both the BEE2_1S and BEE2_2S for assignment of the values accordingly.

# BEE2_1S

# train_data_dc = np.zeros((35374, 8100))
# test_data_dc  = np.zeros((11863, 8100))
# valid_data_dc = np.zeros((9002, 8100))

# BEE2_2S

# train_data_dc = np.zeros((27778, 8100))
# test_data_dc  = np.zeros((9521, 8100))
# valid_data_dc = np.zeros((16192, 8100))

train_target_dc = None
test_target_dc  = None
valid_target_dc = None

def reshape_bee_aux(bee_data, bee_data_dc):
    '''auxiliary function to reshape bee data for sklearn.'''
    for i in range(len(bee_data)):
        bee_data_dc[i] = bee_data[i][0].reshape((8100,))

def reshape_bee_data():
    '''reshape all data for sklearn.'''
    global train_data
    global train_data_dc
    global test_data
    global test_data_dc
    global valid_data
    global valid_data_dc
    reshape_bee_aux(train_data, train_data_dc)
    reshape_bee_aux(test_data,  test_data_dc)
    reshape_bee_aux(valid_data, valid_data_dc)

def reshape_bee_target(bee_data):
    '''reshape bee target given data.'''
    return np.array([np.argmax(bee_data[i][1])
                    for i in range(len(bee_data))])

def reshape_bee_target2(bee_data):
    '''another function for reshaping bee target given data.'''
    return np.array([bee_data[i][1] for i in range(len(bee_data))])

def prepare_bee_data():
    '''reshape and prepare bee data for sklearn.'''
    global train_data
    global test_data
    global valid_data
    reshape_bee_data()

    for i in range(len(train_data)):
        assert np.array_equal(train_data[i][0].reshape((8100,)),
                              train_data_dc[i])

    for i in range(len(test_data)):
        assert np.array_equal(test_data[i][0].reshape((8100,)),
                              test_data_dc[i])

    for i in range(len(valid_data)):
        assert np.array_equal(valid_data[i][0].reshape((8100,)),
                              valid_data_dc[i])

def prepare_bee_targets():
    '''reshape and prepare bee targets for sklearn.'''
    global train_target_dc
    global test_target_dc
    global valid_target_dc    
    train_target_dc = reshape_bee_target(train_data)
    test_target_dc  = reshape_bee_target2(test_data)
    valid_target_dc = reshape_bee_target2(valid_data)

# Individual Decision Tree Classifier
# Here it is used only for testing the decision tree working

def test_dt():
    clf = tree.DecisionTreeClassifier(random_state=0)
    dtr = clf.fit(train_data_dc,train_target_dc)
    print('Decision Tree Training Completed...')

    valid_preds = dtr.predict(valid_data_dc)
    print(metrics.classification_report(valid_target_dc,valid_preds))
    cm1 = confusion_matrix(valid_target_dc,valid_preds)
    print(cm1)

    test_preds = dtr.predict(test_data_dc)
    print (metrics.classification_report(test_target_dc,test_preds))
    cm2 = confusion_matrix(test_target_dc,test_preds)
    print(cm2)

    train_preds = dtr.predict(train_data_dc)
    print(metrics.classification_report(train_target_dc,train_preds))
    cm3 = confusion_matrix(train_target_dc,train_preds)
    print(cm3)

# Ensemble of the Decision Trees which is called as the Random Forest Classifier

def test_rf(num_trees):
    rs = random.randint(0, 1000)
    clf = RandomForestClassifier(n_estimators=num_trees,random_state=rs)
    rf = clf.fit(train_data_dc,train_target_dc)
    print('Random Forest Training Completed...')

    # The below commented code is used to the load the classifiers
    # rf = load('rf_BEE2_1S_'+str(num_trees))
    # rf = load('rf_BEE2_2S_'+str(num_trees))

    # For classification report on the validation data using the metrics.classification_report function
    # Also, the confusion matrix is also attained using the confusion_matrix function
    valid_preds = rf.predict(valid_data_dc)
    print(metrics.classification_report(valid_target_dc,valid_preds))
    cm1 = confusion_matrix(valid_target_dc,valid_preds)
    print(cm1)

    # For classification report on the testing data using the metrics.classification_report function
    # Also, the confusion matrix is also attained using the confusion_matrix function
    test_preds = rf.predict(test_data_dc)
    print(metrics.classification_report(test_target_dc,test_preds))
    cm2 = confusion_matrix(test_target_dc,test_preds)
    print(cm2)

    # For classification report on the training data using the metrics.classification_report function
    # Also, the confusion matrix is also attained using the confusion_matrix function
    train_preds = rf.predict(train_data_dc)
    print(metrics.classification_report(train_target_dc,train_preds))
    cm3 = confusion_matrix(train_target_dc,train_preds)
    print(cm3)

    # The below commented code is used to the save the classifiers
    # save(rf,'rf_BEE2_1S_' + str(num_trees))
    # save(rf,'rf_BEE2_2S_' + str(num_trees))

# This test_rf_range function is used to called the test_rf function on a range of number of decision trees.
def test_rf_range(low_nt, high_nt):
    for i in range (low_nt, high_nt+1):
        print('The number of decision trees in the random forest are ',i)
        test_rf(i)

# This test_rf_range function is used to called the test_rf function on a list of number of decision trees.
def test_rf_list(num_trees_list):
    for num_trees in num_trees_list:
        print('The number of decision trees in the random forest are ',num_trees)
        test_rf(num_trees)

# if __name__ == '__main__':
#     prepare_bee_data()
#     prepare_bee_targets()
#     test_rf_list([5,20,40,50,60,80,100,150,200,250,300])