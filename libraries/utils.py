#------------------------------------------------------#
# Import librairies
#------------------------------------------------------#

import datetime
import hashlib
import os
import time
import urllib

import cv2 as cv
import numpy as np
import pafy
import pandas as pd
import streamlit as st
import wget
import youtube_dl
from imutils.video import FPS, FileVideoStream, WebcamVideoStream
from PIL import Image

import libraries.plugins as plugins

colorWhite = (255, 255, 255)
colorBlack = (0, 0, 0)
colorRed = (255, 0, 0)
colorGreen = (0, 255, 0)
colorBlue = (0, 0, 255)
fontFace = cv.FONT_HERSHEY_SIMPLEX
thickText = 1

#------------------------------------------------------#
# Classes definition
#------------------------------------------------------#


class GUI():
    """
    This class is dedicated to manage to user interface of the website. It contains methods to edit the sidebar for the selected application as well as the front page.
    """

    def __init__(self):

        self.list_of_apps = [
            'Empty',
            'Amenity Detection',
            ]
        self.guiParam = {}

    # ----------------------------------------------------------------

    def getGuiParameters(self):
        self.common_config()
        self.appDescription()
        return self.guiParam

    # ------------------------------------a----------------------------

    def common_config(self, title='Dashboard '): #(Beta version :golf:)
        """
        User Interface Management: Sidebar
        """
        # st.image("./media/logo_inveesion.png","InVeesion.", width=50)

        st.title(title)

        st.sidebar.markdown("### :arrow_right: Settings")

        # Get the application type from the GUI
        self.appType = 'Image Applications'

        self.dataSource = st.sidebar.radio(
                'Please select the source of your ' + self.appType, ['Use Demo Images', 'Upload image from local machine', 'Image URL'])


        # Get the application from the GUI
        self.selectedApp = st.sidebar.selectbox(
            'Chose an AI Application', self.list_of_apps)

        if self.selectedApp is 'Empty':
            st.sidebar.warning('Please select Amenity Detection above')

        self.displayFlag = st.sidebar.checkbox(
            'Display Real-Time Results', value=True)

        # Update the dictionnary
        self.guiParam.update(
            dict(selectedApp=self.selectedApp,
                 appType=self.appType,
                 dataSource=self.dataSource,
                 displayFlag=self.displayFlag))

    # -------------------------------------------------------------------------

    def appDescription(self):

        st.header(' :arrow_right: Application: {}'.format(self.selectedApp))

        if self.selectedApp == 'Amenity Detection':
            st.info(
                'This web application performs object detection using advanced deep learning models. It can detect 30 classes of common household objects from OpenImages dataset.')
            self.sidebarAmenityDetection()

        else:
            st.info(
                'To start using this web application you must first select Amenity Detection from the sidebar menu.')

    # --------------------------------------------------------------------------
    def sidebarEmpty(self):
        pass
    # --------------------------------------------------------------------------


    def sidebarAmenityDetection(self):

        st.sidebar.markdown("### :arrow_right: Model")
        #------------------------------------------------------#
        model = st.sidebar.selectbox(
            label='Select the model',
            options=['Default_YOLOv4', 'Fine_tuned_YOLOv4', 'Fine_tuned_YOLOv4_tiny'])

        st.sidebar.markdown("### :arrow_right: Model Parameters")
        #------------------------------------------------------#
        confThresh = st.sidebar.slider(
            'Select the Confidence threshold', value=0.3, min_value=0.0, max_value=1.0)
        nmsThresh = st.sidebar.slider(
            'Select the Non-maximum suppression threshold', value=0.30, min_value=0.0, max_value=1.00, step=0.05)

        self.guiParam.update(dict(confThresh=confThresh,
                                  nmsThresh=nmsThresh,
                                  model=model,
                                  #   desired_object=desired_object
                                  ))

# ------------------------------------------------------------------
# ------------------------------------------------------------------


class AppManager:
    """
    This is a master class
    """

    def __init__(self, guiParam):
        self.guiParam = guiParam
        self.selectedApp = guiParam['selectedApp']

        self.model = guiParam['model']
        self.objApp = self.setupApp()

    # -----------------------------------------------------

    def setupApp(self):
        """
        #
        """

        if self.selectedApp == 'Amenity Detection':

            if self.model == 'Default_YOLOv4':

                self.paramDefaultYOLOv4 = dict(labels='models/default_yolov4/coco.names',
                                          modelCfg='models/default_yolov4/yolov4.cfg',
                                          modelWeights="models/default_yolov4/yolov4.weights",
                                          confThresh=self.guiParam['confThresh'],
                                          nmsThresh=self.guiParam['nmsThresh'])

                self.objApp = plugins.Object_Detection_YOLO(self.paramDefaultYOLOv4)

            elif self.model == 'Fine_tuned_YOLOv4':

                self.paramFineTunedYOLOv4 = dict(labels='models/amenity_yolov4/amenity.names',
                                          modelCfg='models/amenity_yolov4/amenity_yolov4.cfg',
                                          modelWeights="models/amenity_yolov4/amenity_yolov4.weights",
                                          confThresh=self.guiParam['confThresh'],
                                          nmsThresh=self.guiParam['nmsThresh'])

                self.objApp = plugins.Object_Detection_YOLO(self.paramFineTunedYOLOv4)

            elif self.model == 'Fine_tuned_YOLOv4_tiny':
                self.paramFineTunedYOLOv4Tiny = dict(labels='models/amenity_yolov4_tiny/amenity.names',
                                          modelCfg='models/amenity_yolov4_tiny/amenity_yolov4_tiny.cfg',
                                          modelWeights="models/amenity_yolov4_tiny/amenity_yolov4_tiny.weights",
                                          confThresh=self.guiParam['confThresh'],
                                          nmsThresh=self.guiParam['nmsThresh'])

                self.objApp = plugins.Object_Detection_YOLO(self.paramFineTunedYOLOv4Tiny)

            else:
                raise ValueError(
                    '[Error] Please selected one of the listed models')


        # -----------------------------------------------------

        else:
            raise Exception(
                '[Error] Please select one of the listed application')

        return self.objApp

    # -----------------------------------------------------
    # -----------------------------------------------------

    def process(self, frame, motion_state):
        """
        # return a tuple: (bboxed_frame, output)
        """
        bboxed_frame, output = self.objApp.run(frame, motion_state)

        return bboxed_frame, output

# ------------------------------------------------------------------
# ------------------------------------------------------------------


class DataManager:
    """
    """

    def __init__(self, guiParam):
        self.guiParam = guiParam

        self.url_demo_images = {
            "Living-Room": "https://a0.muscache.com/im/pictures/6080d6b0-1cb6-4f44-9776-1892e99c9c81.jpg?aki_policy=xx_large",
            "Living-Room-2":"https://i.pinimg.com/originals/ed/74/3f/ed743f301349446587428a5553a4ce56.jpg",
            "Wine-rack" : "https://imagesvc.meredithcorp.io/v3/mm/image?q=85&c=sc&poi=face&w=1871&h=980&url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F9%2F2019%2F01%2Fwine-racks-main-HP-BLOG0220.jpg",
            "Kitchen": "https://hgtvhome.sndimg.com/content/dam/images/hgtv/fullset/2018/4/23/1/HUHH2018-Curb-Appeal_Seattle-WA_11.jpg.rend.hgtvcom.966.644.suffix/1524514638493.jpeg",
            "Bathroom": "https://www.thespruce.com/thmb/5u0qWkAXxlIFM2kYS73Db_4ZImE=/1000x562/smart/filters:no_upscale()/smallbathroomwithgreattextures-6cf9756c3ede44969e02b60f7ad7242d.jpg"}

        self.demo_image_examples = {"Bedroom": "./data/demo-bedroom.jpg",
                                    "Coffeemaker": "./data/demo-coffee.jpg",
                                    "Kitchen": "./data/demo-kitchen.jpg",
                                    "Pool": "./data/demo-pool.jpg",
                                    "Living_room": "./data/demo-living.png",}

        self.image = None
        self.data = None

  #################################################################
  #################################################################

    def load_image_source(self):
        """
        'Use Demo Images', 'Upload image from local machine', 'Image URL'
        """

        if self.guiParam["dataSource"] == 'Use Demo Images':

            @st.cache(allow_output_mutation=True)
            def load_image_from_path(image_path):
                image = cv.imread(image_path, cv.IMREAD_COLOR)
                # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                return image

            file_path = st.text_input('Enter the image PATH')

            if os.path.isfile(file_path):
                self.image = load_image_from_path(image_path=file_path)

            elif file_path is "":
                file_path_idx = st.selectbox(
                    'Or select a demo image from the list', list(self.demo_image_examples.keys()))
                file_path = self.demo_image_examples[file_path_idx]

                self.image = load_image_from_path(image_path=file_path)
            else:
                raise ValueError("[Error] Please enter a valid image path")

            #--------------------------------------------#
            #--------------------------------------------#

        elif self.guiParam["dataSource"] == 'Upload image from local machine':

            @st.cache(allow_output_mutation=True)
            def load_image_from_upload(file):
                tmp = np.fromstring(file.read(), np.uint8)
                return cv.imdecode(tmp, 1)

            file_path = st.file_uploader(
                'Upload an image', type=['png', 'jpg'])

            if file_path is not None:
                self.image = load_image_from_upload(file_path)
            elif file_path is None:
                raise ValueError(
                    "[Error] Please upload a valid image ('png', 'jpg')")
            #--------------------------------------------#
            #--------------------------------------------#

        elif self.guiParam["dataSource"] == 'Image URL':

            @st.cache(allow_output_mutation=True)
            def load_image_from_url(url_image):
                """
                """
                resp = urllib.request.urlopen(url_image)
                tmp = np.asarray(bytearray(resp.read()), dtype="uint8")
                return cv.imdecode(tmp, cv.IMREAD_COLOR)

            file_path = st.text_input('Enter the image URL')

            if file_path is not "":
                self.image = load_image_from_url(url_image=file_path)

            elif file_path is "":

                file_path_idx = st.selectbox(
                    'Or select a URL from the list', list(self.url_demo_images.keys()))
                file_path = self.url_demo_images[file_path_idx]

                self.image = load_image_from_url(url_image=file_path)
            else:
                raise ValueError("[Error] Please enter a valid image URL")

            #--------------------------------------------#
            #--------------------------------------------#

        else:
            raise ValueError("Please select one source from the list")

        return self.image


    def load_image_or_video(self):
        """
        Handle the data input from the user parameters
        """
        if self.guiParam['appType'] == 'Image Applications':
            self.data = self.load_image_source()

        else:
            raise ValueError(
                '[Error] Please select of the two Application pipelines')

        return self.data
