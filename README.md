# openvino_colab
Intel OpenVINO AI framework examples using Google Colab.

Loosely based on  [OpenDevLibrary](https://github.com/alihussainia/OpenDevLibrary) by [Alihussainia](https://github.com/alihussainia/)

# interview_prep.ipynb
Interview preparation app leveraging pre-trained OpenVINO models. 

## [Human Pose Estimation](https://docs.openvinotoolkit.org/2020.1/_models_intel_human_pose_estimation_0001_description_human_pose_estimation_0001.html)

The human pose estimation model can be used to detect if a correct lean forward posture is used on interview practice.

## [Face Detection](https://docs.openvinotoolkit.org/2019_R1/_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)

The face detection model can be used to detect where in the image/video are the faces

## [Emotions Recognition](https://docs.openvinotoolkit.org/2020.1/_models_intel_emotions_recognition_retail_0003_description_emotions_recognition_retail_0003.html#outputs)  
  
Face detection output can be used as input to recognize face emotions. Emotion recognition can be used to try and detect if you appear happy, surprised or neutral as opposed to sad or angry.

## Running the Project
Follow the steps on the [interview_prep.ipynb notebook](https://github.com/socd06/openvino_colab/blob/master/interview_prep.ipynb)

## Command Line Parameters
Change the -m models parameter depending on the desired output 

## demo.ipynb
Notebook showing how to integrate OpenVINO with Google Colab

## emotions.ipynb
Emotions classifier using OpenVINO in Google Colab

## License
[MIT License](https://github.com/socd06/openvino_colab/blob/master/LICENSE)
