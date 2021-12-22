# coding: utf-8

# In[13]:


'''
CS585 - Assignment 4 Part 1

Tracking Bats in flight

The goal of this part of the assignment was to track moving bats against a background. 
We employed the Kalman filter and a greedy algorithm to perform the data analysis. 


Leyan Abdulhamid - U06764115

Teammate: Shalei Kumar - U12668241

'''


import numpy as np
import math
import sys
import cv2
import pandas as pd
import os
import random
import warnings
random.seed(1)





'''
From: http://cs-people.bu.edu/sjzhao/CS585/A4/code/HW4_TrackingBats/kalman_filter.py
'''


def speed(dataset):
    ''' Calculates the difference in x and y values of the object between images and adds a deltaX and detlaY column to the df '''
    deltaX = [0]
    deltaY = [0]
    for i in range(1, len(dataset)):
        deltaX.append(dataset['x'][i] - dataset['x'][i-1])
        deltaY.append(dataset['y'][i] - dataset['y'][i-1])
    dataset['deltaX'] = deltaX
    dataset['deltaY'] = deltaY
    return dataset



def kalmanFilter(image_frames, measurements):
    '''
    takes in image frames and measurements, make predictions based on previous prediction and current measurements. 
    Return a list of tracks of objects between frames, from previous frame to this frame.

    '''
    img = image_frames[0]
    img_x, img_y = img.shape[:2]
    
    borderDistance = 50

    predictions = []
    firstMeasurement = measurements[0]
    firstMeasurement = speed(firstMeasurement)
    length, noStates = firstMeasurement.shape

    deltaT = 1

    # state transition model
    A = np.array([[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Q : PROCESS NOISE COVARIANCE
    # Generally constructed intuitively
    # if confident about prediction set Q to 0.
    # Example: Q = eye(n) n: # states, x, y
    Q = np.eye(noStates) # * 0.1

    # external influence B and vector u UNKNOWN

    # shape transformation vector trivial
    H = np.eye(noStates)

    # R: MEASUREMENT NOISE COVARIANCE MATRIX
    # should be 0 since using ground truth
    R = np.eye(noStates) * 0.1 # * 0.01

    I = np.eye(noStates)

    # process error covariance
    # many times P is diagonal, how much deviation you expect in the initialization
    # if dont know how to start use identity matrix
    # first p_k set to R
    p_k = Q

    # first x_k set to firstmeasurement 
    x_k = firstMeasurement.values
    for i in range(1, len(measurements)):#len(measurements) #, len(locFiles
        x_k_prev = x_k
        p_k_previous = p_k
        measurement = measurements[i]
        measurement = speed(measurement)

        x_k_predict = np.dot(x_k_prev, A) # + w_k
        p_k_predict = np.dot(np.dot(A, p_k_previous), A.T) # + Q
        p_k_predict = p_k_predict * np.eye(noStates)
        
        # data association
        # res is the INDEX OF MEASUREMENTS associated to the predict value
        # Make data association only based on X and Y, left out deltaX, deltaY
        res = data_association(x_k_predict[:, :2], measurement.values[:, :2])
        res = np.array(res)

        # check if there are more predictions than measurements
        # check if res contains None: indexes of prediction that does not have a measurement
        # sort index inreverse
        indexes = np.where(np.array(res) == None)[0][::-1]
        # if object has no measurement
        for index in indexes:
            # if object out of the frame # SPEED explodes later, so cannot evaluate based on speed            
            if (((x_k_predict[index][0]) >= img_x-borderDistance) 
            or ((x_k_predict[index][0]) <= borderDistance)
            or ((x_k_predict[index][1]) >= img_y-borderDistance)
            or ((x_k_predict[index][1]) <= borderDistance)):
                # drop the prediction
                res = np.delete(res, index)
                x_k_predict = np.delete(x_k_predict, index, axis = 0)
                x_k_prev = np.delete(x_k_prev, index, axis = 0)
            # else if occlusion happens
            # remove the unassigned measurement
            else:
                res = np.delete(res, index)
                x_k_predict = np.delete(x_k_predict, index, axis = 0)
                x_k_prev = np.delete(x_k_prev, index, axis = 0)

        y = measurement.values[res.tolist()]

        # Kalman gain
        np.warnings.filterwarnings('ignore')
    
        K = p_k_predict/(R + p_k_predict)
        K = np.nan_to_num(K, 0)
       
        # update, reconcile
        x_k = x_k_predict+ np.dot((y - x_k_predict), K)

        # process covariance mat update
        p_k = (I - K)* p_k_predict

        prev_points = x_k_prev[:, :2].astype(int)
        present_points = x_k[:, :2].astype(int)

        predictions.append([prev_points, present_points])

        #check if there are more measurements than prediction and add measurement to prediction
        if len(measurement) > len(x_k):
            for i in range(len(measurement)):
                if i not in res:
                    x_k = np.append(x_k, measurement.values[i].reshape(1, 4), axis = 0)
    return predictions









def costMatrix(A, B):
    ''' Calculate the cost matrix of two sets of points '''
    cost_matrix = [None] * len(A)
    for i in range(len(A)):
        cost_matrix[i] = [None] * len(B)
        for j in range(len(B)):
            cost_matrix[i][j] = math.sqrt((A[i][0] - B[j][0]) ** 2 + (A[i][1] - B[j][1]) ** 2)
    return cost_matrix





def greedyAlgorithm(matrix):
    ''' finds the best prediction value (the smallest value distance of that prediction) ''' 
    measures = [None] * len(matrix)
    for i in range(len(matrix)):
        smallest_val = sys.maxsize
        for j in range(len(matrix[i])):
            if matrix[i][j] < 300 and matrix[i][j] < smallest_val and j not in measures:
                measures[i] = j
                smallest_val = matrix[i][j]
    return measures



def data_association(predictions, measures):
    ''' finds the optimal solution using the cost matrix and greedy algorithm '''     
    cost = costMatrix(predictions, measures)
    greedy_res = greedyAlgorithm(cost)
    return greedy_res




def bat_track(frames, track_info):
    ''' Draws lines of the bat's predicted movement '''
    result_frames = frames.copy()
    def drawLine(track_index, pos_index, color):
        if track_info[track_index][0][pos_index][0] == -1:
            return
        
        trace_end_index = min(track_index + 10, len(frames))
        for frame_index in range(track_index+1, trace_end_index):
            cv2.line(result_frames[frame_index], tuple(np.array(track_info[track_index][0][pos_index]).astype(int)), tuple(np.array(track_info[track_index][1][pos_index]).astype(int)), color, 2)
        
        track_info[track_index][0][pos_index] = [-1, -1]

        if track_index == len(track_info) - 1:
            track_info[track_index][1][pos_index] = [-1, -1]
            return
            
        for next_pos_index in range(len(track_info[track_index+1][0])):
            if track_info[track_index+1][0][next_pos_index][0] != -1 and np.all(track_info[track_index][1][pos_index] == track_info[track_index+1][0][next_pos_index]):
                drawLine(track_index+1, next_pos_index, color)
                track_info[track_index][1][pos_index] = [-1, -1]
                break
    
    for i in range(len(track_info)):
        for j in range(2):
            for k in range(len(track_info[i][j])):
                if track_info[i][j][k][0] != -1:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    drawLine(i, k, color)
    
    return result_frames



# store the frames in a directory
def save_images(dir, frames):
    # write the image to the output file
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    count = 1
    for output in frames:
        cv2.imwrite(dir+str(count)+'.jpeg',output)
        count += 1

    # Make a video
    out = cv2.VideoWriter(dir+'TrackBats_Result_Video.mp4', cv2.VideoWriter_fourcc(*'XVID'), 15, (700,700))
    for img in frames:
        img = cv2.resize(img, (700,700))
        out.write(img)
    out.release()





def main():
    # load data, parameter types refer to preprocessing.py
    img_dir=r'C:\Users\leyan\Documents\CS585\PA4\CS585-BatImages\Gray'
    loc_dir=r'C:/Users\leyan\Documents\CS585\PA4\Localization_Bats'

    loc_path = os.path.abspath(loc_dir)
    loc_list = [os.path.join(loc_path, file) for file in os.listdir(loc_path) if os.path.isfile(os.path.join(loc_path, file)) and 'txt' in file]
    loc_list.sort()

    image_path = os.path.abspath(img_dir)
    image_list = [os.path.join(image_path, file) for file in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, file)) and 'ppm' in file] 
    image_list.sort()

    image_frames = []
    for file in image_list:
        img = cv2.imread(file)
        image_frames.append(img.copy())
    image_frames = np.array(image_frames)

    loc_data = []
    for file in loc_list:
        dataframe = pd.read_csv(file, names=['x', 'y'])
        loc_data.append(dataframe.copy())
    

    lines = kalmanFilter(image_frames, loc_data)
    
    line_frames = bat_track(image_frames, lines)
    save_images('C:/Users/leyan/Documents/CS585/PA4/TrackBat_Results/', line_frames)

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    main()

