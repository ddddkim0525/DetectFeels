# DetectFeels
Detecting Emotions from given pictures

1. Face API
  A short python script to send kaggle dataset pictures to Face API. Training Results are sotred in prediction.txt.
  Labeling: -1 if error, 0~6 refer to original mapping in the kaggle dataset.
  
  Face API performed 70% accuracy on the first 1100 photos of the small kaggle dataset.
  
  kaggle dataset : https://www.kaggle.com/c/facial-keypoints-detector
  
2. Face Detection and Extraction
  Module that detects and extract faces from the picture. Saves the transformed file. 
  There are quite a lot of dependencies, please use the openface.yml to setup the conda environment.

3. Emotion Prediction
  Performs transfer learning to detect emotions. Using tensornets to load existing architecture and weights.
  Please install tensorflow and tensernet to run the script.
  Currently only VGG19 implemented. Adding new architectures should be simple using tensornet.
  tensorflow.yml included just incase, but advised not to, its very messy as its my main environment.
  
  Uses the Face recognition dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
  
  dataset above is not included in the git as the size is too big. please download it.
  
  
  References:
  
  https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
  
  http://tamaszilagyi.com/blog/2019/2019-01-12-tensornets/
