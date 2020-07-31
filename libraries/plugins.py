
import os
import time
from collections import Counter

import cv2 as cv
import numpy as np
import pafy
import pandas as pd
import streamlit as st

colorWhite = (255, 255, 255)
colorBlack = (0, 0, 0)
colorRed = (255, 0, 0)
colorGreen = (0, 255, 0)
colorBlue = (0, 0, 255)
fontFace = cv.FONT_HERSHEY_SIMPLEX
thickText = 1


def thickRect(H):
    return int(round(H/150))


#################################################################################
#################################################################################
# Object Detection
#################################################################################
#################################################################################


class Object_Detection_YOLO:

    def __init__(self, paramYolo):
        """
        # Initialize parameters
        """

        # Confidence and NMS thresholds
        self.confThreshold = paramYolo['confThresh']  # 0.5
        self.nmsThreshold = paramYolo['nmsThresh']  # 0.3
        self.inpWidth = 416  # Width of network's input image
        self.inpHeight = 416  # Height of network's input image
        self.classes = self.getClasses(paramYolo['labels'])
        try:
            self.colors = paramYolo['colors']

        except:
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Load YOLO model
        self.modelCfg = paramYolo['modelCfg']
        self.modelWeights = paramYolo['modelWeights']
        self.net = self.load_YOLO_model(
            self.modelCfg, self.modelWeights)

        # Display results
        self.output_result = dict()

        self.allNumber_of_detections = []
        self.allDetected_object = []
        self.allConfidence = []
        self.saveConfidences = []
        self.saveClassId = []

 ##########################################################################
    # Define some methods
 ##########################################################################

    @st.cache(allow_output_mutation=True)
    def load_YOLO_model(self, modelCfg, modelWeights):
        """
        # Load YOLO model
        """
        if os.path.exists(modelCfg) and os.path.exists(modelWeights):
            net = cv.dnn.readNetFromDarknet(
                cfgFile=modelCfg, darknetModel=modelWeights)
        else:
            st.error('One or both files does not exist')

        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        return net

    # -------------------------------------------------------------------------

    def getClasses(self, classesFile):
        """
        # Load names of classes
        """
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        return classes

    # -------------------------------------------------------------------------

    def getOutputsNames(self, net):
        """
        Get the names of the output layers
        """

        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()

        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # -------------------------------------------------------------------------

    def draw_bbox(self, frame, classId, conf, box, color):
        """
        # Draw the predicted bounding box
        """

        # Draw a bounding box.
        left, top, right, bottom = box
        cv.rectangle(img=frame, pt1=(left, top), pt2=(right, bottom), color=color, thickness=3)

        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '{}:{:.2f}'.format(self.classes[classId], conf*100)

        # Display the label at the top of the bounding box
        (label_width, label_height), baseLine = cv.getTextSize(
            text=label, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.45, thickness=thickText)

        top = max(top, label_height)
        # print(labelSize, baseLine)

        cv.rectangle(frame, pt1=(left, top - round(1.5*label_height)), pt2=(left + round( 1.5 * label_width), top + baseLine), color=colorWhite, thickness=cv.FILLED)
        cv.putText(img=frame, text=label, org=(left, top), fontFace=fontFace, fontScale=0.65, color=colorBlack, thickness=thickText)

    # -------------------------------------------------------------------------

    def postprocess(self, frame, outs):
        """
        # Remove the bounding boxes with low confidence using non-maxima suppression
        """
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []

        self.saveConfidences = []
        self.saveClassId = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    # save results
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        indices = cv.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            box = [left, top, left + width, top + height]

            self.draw_bbox(frame, classIds[i],
                           confidences[i], box, self.colors[classIds[i]])

            self.saveConfidences.append(confidences[i])
            self.saveClassId.append(self.classes[classIds[i]])

    # -------------------------------------------------------------------------

    def run(self, frame, motion_state):
        """
        # Call this process method for each frame to perfor detection using YOLO
        """
        tic = time.time()
        bboxed_frame = np.copy(frame)
        if motion_state:
            ###

            # Create a 4D blob from a frame.
            blob = cv.dnn.blobFromImage(
                frame, 1/255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)

            # Sets the input to the network
            self.net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = self.net.forward(self.getOutputsNames(self.net))

            # Remove the bounding boxes with low confidence
            self.postprocess(bboxed_frame, outs)

        else:
            pass

        # Keep tracks of results
        self.allNumber_of_detections.append(len(self.saveClassId))
        self.allDetected_object.append(dict(Counter(self.saveClassId)))
        self.allConfidence.append(
            100*np.around(self.saveConfidences, decimals=2))

        # Create a dataframe
        dataframe_plugin = pd.DataFrame(data=list(zip(
            self.allNumber_of_detections,
            self.allDetected_object,
            self.allConfidence)),
            columns=["Nbr Det Objects", "Det. Objects", "Confid."])

        # Update th results dictionnary
        self.output_result.update(
            dict(time_per_frame=time.time() - tic,
                 detected_object=self.allDetected_object,
                 number_of_detections=self.allNumber_of_detections,
                 confidence=self.allConfidence,
                 dataframe_plugin=dataframe_plugin))

        return bboxed_frame, self.output_result

    # -------------------------------------------------------------------------

    def display_summary(self, ph, output_result):
        """
        # Display results
        """

        if output_result["displayFlag"]:

            ph[0].markdown('### Processing Results')

            ph[1].markdown(
                '* Number of detected objects :\t {}'.format(output_result['number_of_detections'][-1]))

            ph[2].markdown(
                '* Detected objects  :\t {}'.format(output_result['detected_object'][-1]))

            ph[3].markdown(
                '* Detections probabilities (%):\t {}'.format(output_result['confidence'][-1]))
        else:
            pass

#################################################################################
#################################################################################