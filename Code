from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib



act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  'Andrea Anders',    'Ashley Benson', 'Christina Applegate',    'Dianna Agron',  'Gillian Anderson']

female = ['anders', 'benson', 'applegate', 'agron', 'anderson']
male = ['eckhart', 'sandler', 'brody']

# Please change to main directory
os.chdir('/Users/emmadaponte/UofT/CSC320/p3')
# Please read comments in __main__() to easily see which folders are needed
# for program to work correctly, and to save outputs

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    '''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            

def pca(X):
    """    Principal Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean.
        From: Jan Erik Solem, Programming Computer Vision with Python
        #http://programmingcomputervision.com/
    """
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(abs(e))[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X



#Note: you need to create the uncropped folder first in order 
#for this to work
def extract():
  """ Taken from Michael Guerzhoy's get_data.py
  http://www.cs.toronto.edu/~guerzhoy/320/proj3/get_data.py
  """
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("faces_subset.txt"):
            print('working')
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long 
                # to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+\
                filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    # If file wasn't saved, still increase i to get correct 
                    # image number
                    i+=1
                    continue
    
                
                print filename
                i += 1
        
# Need to have 'cropped' folder to work
def crop():
    ''' Crop the faces according to the coordinates given in faces_subset.txt
    '''
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("faces_subset.txt"):
            print('working')
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                if not os.path.isfile("uncropped/"+filename):
                    # Not all images were downloaded, need to skip some numbers
                    i+=1
                else:
                    print(filename)
                    try:
                        orig = imread('uncropped/'+filename)
                        coords = line.split()[5].split(',')
                        x1 = int(coords[0])
                        x2 = int(coords[2])
                        y1 = int(coords[1])
                        y2 = int(coords[3])
                        
                        # Not all images have 3 color channels
                        if len(orig.shape)==3:
                            cropped = orig[y1:y2,x1:x2,:]
                        else:
                            cropped = orig[y1:y2,x1:x2]
                        
                        imsave("cropped/"+filename, cropped)
                        
                        i+=1
                    except:
                        i+=1
                        
# Need folder 'resized' to work
def res():
    ''' Resize the faces in folder 'cropped' to be 32 x 32 
    '''
    for file in os.listdir("cropped/"):
        try:
            orig = imread("cropped/"+file)
            
            while orig.shape[0]>64:
                 orig = imresize(orig, .5)
                
            resized = imresize(orig, [32,32])
            
            grayed = 0.299*resized[:,:,0] + 0.587*resized[:,:,1] +\
             0.114*resized[:,:,2]
            
            imsave("resized/"+file, gray)
        except:
            continue

# Need to have folders called 'validation', 'training', and 'test' to work
def sep():
    ''' Separates images from 'resized' folder into three independent sets:
    Training, Validation, and Test
    '''
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for filename in os.listdir("resized/"):
            if name in filename and i<120:
                to_move = imread("resized/"+filename)
                if i<100:
                    imsave("training/"+filename, to_move)
                elif i<110:
                    imsave("validation/"+filename, to_move)
                elif i<120:
                    imsave("test/"+filename, to_move)
                i+=1
            else:
                continue
                

def make_T():
    ''' Creates a matrix where each column is a flattened image taken from 
    the training set created by sep(). Contains 800 flattened images with 
    1024 entries per image.
    
    return: the matrix
    '''
    i = 0
    T = zeros([800, 1024])
    
    for file in os.listdir("training/"):
        try:
            image = imread("training/"+file)
            add = image[:,:,0]
            add = add/255.0
            add = add.flatten()
            T[i,:] = add
            i+=1
        except:
            continue
    return T
    
def generate(training_matrix):
    ''' Saves the first 25 eigenfaces and the average face, for use with report
    
    input: training_matrix (matrix with flattened training faces for columns)
    '''
    proj, var, mean = pca(training_matrix)
    mean = mean.reshape(32,32)
    imsave("outputs-part2/average_face.jpg", mean)
    i=0
    while i<25:
        to_save = proj[i].reshape(32,32)
        imsave("outputs-part2/eigenface_"+str(i), to_save)
        i+=1
        
        
def project(input, eig_matrix, k):
    ''' Projects an image, input, onto the first k eigenfaces as found in
     the matrix of eigenfaces, eig_matrix
    
    input: input (32x32 grayscale image), eig_matrix (projection matrix as
     given by pca()), k (int, number of eigenfaces to project onto)
    
    return: list of size k with values given by individually projecting input 
    onto first k eigenfaces
    '''
    n = 0
    vals = zeros(k)
    input = input.reshape(1024,1)
    while n < k:
        eig = eig_matrix[n]
        vals[n] = dot(eig.T, input)
        n += 1
        
    return vals
            
# Opening and projecting all of the training images is computationally expensive
# This does it once for each value k and saves the information for later use
# Need folder called 'vals' to store data
def proj_training(eig_matrix):
    ''' Projects every image in the training set onto a given number of eigen-
    faces as found in eig_matrix. Saves the values given by this to a file with
    same name as the training image that was projected. 
    
    Projects each image onto 2, 5, 10, 20, 50, 80, 100, 150, and finally 200
    eigenfaces. Values recieved from each projection are saved in lists on 
    newlines in textfile so that they may easily be accessed later.
    
    input: eig_matrix (projection matrix as given by pca())
    '''
    for file in os.listdir("training/"):
        try:
            image = imread("training/"+file)
            image = image[:,:,0]
            image = image/255.0
            for k in [2, 5, 10, 20, 50, 80, 100, 150, 200]:
                im_vals = project(image, eig_matrix, k)
                new_file_name = file.split('.')[0]
                new_file = open("vals/"+new_file_name + ".txt", 'a')
                new_file.write(str(im_vals[0]))
                for val in im_vals[1:]:
                    new_file.write(", "+str(val))
                new_file.write("\n")
                new_file.close()
        except:
            continue
            

def proj_compare(input, eig_matrix, k):
    ''' Projects input, 32x32 grayscale image, onto the first k values of
    eig_matrix, as given by make_T(), and then compares this to the projection
    of all training images over the first k eigenfaces to find the most similar
    image (the one whose projection is closest to the input's projection)
    
    input: input (32x32 grayscale face image), eig_matrix (projection matrix
    as given by pca()), k (int, number of eigenfaces to project onto)
    return: filename of training image with closest projection 
    '''
    distance = inf
    closest = 'None'
    input = input[:,:,0]
    input = input/255.0
    in_vals = project(input, eig_matrix, k)
    for file in os.listdir("training/"):
        try:
            training_vals = open('vals/'+file.split('.')[0] + '.txt')
            all_vals = training_vals.read()
            all_vals = all_vals.split('\n')
            all_vals = all_vals[:-1]
            
            for i in range(len(all_vals)):
                all_vals[i] = all_vals[i].split(',')
                for j in range(len(all_vals[i])):
                    all_vals[i][j] = float(all_vals[i][j])
            # find correct k comparisons
            for values in all_vals:
                if len(values) == k:
                    im_vals = values
            
            if norm(in_vals-im_vals) < distance:
                distance = norm(in_vals-im_vals)
                closest = file
        except:
            continue
            
    return closest

# Saves first 5 failed matches as seen in report. Need 'outputs-part3' folder
def results(folder, k_vals=[2, 5, 10, 20, 50, 80, 100, 150, 200]):
    ''' Gives results for matching all images in a given folder, respective to
    how many eigenfaces were used. The number of eigenfaces used is given by
    k_vals, which has a list of values, so that we may determine which value in 
    k_vals has the highest match success rate 
    
    input: folder (folder containing images we wish to match), k_vals (list of
    ints, match images using k eigenfaces, for k in k_vals, and record results)
    
    return: names (list of lists, each list has how many eigenfaces were used
    (K) in index 0, number of successive name matches in index 1, number of 
    failed matches in index 2), genders (list of lists, same information as 
    names, but records gender matches rather than name matches)
    '''
    T = make_T()
    proj = pca(T)[0]
    
    names = zeros([len(k_vals),3])
    genders = zeros([len(k_vals), 3])
    
    round = 0
    fails = 0
    
    for file in os.listdir(folder):
        print('Round: ' + str(round))
        round+=1
        try:
            input = imread(folder+"/"+file)
            it = 0
            for a in act:
                a = a.split()[1].lower()
                if a in file:
                    input_name = a
                    
            for k in k_vals:    
                found_file = proj_compare(input, proj, k)
                for a in act:
                    a = a.split()[1].lower()
                    if a in found_file:
                        found_name = a
                if found_name == input_name:
                    names[it] = [k, names[it][1] + 1, names[it][2]]
                else:
                    names[it] = [k, names[it][1], names[it][2] + 1]
                    # Display first 5 failures report, need 'outputs-part3'
                    if (k==50 and fails < 5):
                        found = imread('training/' +found_file)
                        both = zeros([32,64,3])
                        both[:,:32,:] = input
                        both[:,32:,:] = found
                        both = both[:,:,0]
                        imsave("outputs-part3/" + file+'-'+found_file+\
                        '.jpg', both)
                        fails +=1
                
                if (input_name in female) and (found_name in female):
                    genders[it] = [k, genders[it][1] + 1, genders[it][2]]
                elif (input_name in male) and (found_name in male):
                    genders[it] = [k, genders[it][1] + 1, genders[it][2]]
                else:
                    genders[it] = [k, genders[it][1], genders[it][2] + 1]
                it += 1
            
        except:
            continue
    
    return [names,genders]
    
    
def find_apply_best_k(save=False, val_file='validation/', test_file='test/'):
    ''' Attempt to match all images in val_file using function results(), then
    access information returned by results to determine how many eigenfaces, k, 
    should be projected onto in order for there to be a maxiumum number of
    successful matches. 
    
    Use this best number of eigenfaces, k, to attempt to match all images in
    test_file, and return the new success/failure results
    
    input:  save (boolean, set to True to save results to file 'results'), 
    val_file (location of validation images), test_file (location of test 
    images)
    
    return: new_name_stats (list embedded in list with results of name matching, 
    index 0 is the k eigenfaces used, index 1 is number of successes, index 2 is 
    number of fails), new_gender_stats (list embedded in list with results of 
    gender matching, same set up as for new_name_stats)
    '''
    print("Finding best k ...\n")
    result = results(val_file)
    print('Table for name recognition: \n' + "K, Successes, Fails\n"\
     + str(result[0]) + '\n' + 'Table for gender recognition: \n' + \
     "K, Successes, Fails \n" +str(result[1]))
    
    if save:
        place = open('results.txt', 'w')
        place.write("    VALIDATION SET RESULTS\n")
        place.write("Names Stats, all k: \n K, Successes, Fails \n")
        place.write(str(result[0]))
        place.write("\nGenders Stats, all k:\n K, Successes, Fails \n")
        place.write(str(result[1]))
        
    print("\n\nApplying best k ...")
    
    name_stats = []
    for stat in result[0]:
        name_stats.append(stat[1])
    where_k1 = name_stats.index(max(name_stats))
    best_k1 = result[0][where_k1][0]
    
    new_name_stats = results(test_file, [best_k1])[0]
    
    print('New name recognition statistics: \n' + "K, Successes, Fails, \n")
    print(str(new_name_stats))
    
    gender_stats = []
    for stat in result[1]:
        gender_stats.append(stat[1])
    where_k2 = gender_stats.index(max(gender_stats))
    best_k2 = result[1][where_k2][0]
    
    new_gender_stats = results(test_file, [best_k2])[1]
    
    print('New gender recognition statistics: \n' +"K, Successes, Fails \n")
    print(str(new_gender_stats))
    
    if save:
        place.write("\n     TEST SET RESULTS")
        place.write("\nNames Stats, best k: \n K, Successes, Fails \n")
        place.write(str(new_name_stats))
        place.write("\nGenders Stats, best k:\n K, Successes, Fails \n")
        place.write(str(new_gender_stats))
        place.close()
        
    return [new_name_stats, new_gender_stats]
    
def __main__():
    gray()
        # Uncomment below if need to download images (need 'uncropped' folder)
    # extract()
    
        # Uncomment below if need to crop images (need 'cropped' folder)
    # crop()
    
        # Uncomment below if need to resize images (need 'resized' folder)
    # res()
    
        # Uncomment below if need to separate images in different sets
        # (need 'training', 'validation', and 'test' folders)
    # sep()
    
    training_matrix = make_T()
    eig_matrix, var, mean = pca(training_matrix)
    
    # Save average face and first 25 eigenfaces (need 'outputs-part2 folder)
    generate(training_matrix)
    
    # Save data for training set projections onto eigenface matrix
    # (need 'vals' folder)
    proj_training(eig_matrix)
    
    # Need 'outputs-part3' folder for first 5 failed matches. Creates 
    # results.txt that saves table of successful matches for given k values
    find_apply_best_k(True)
    
