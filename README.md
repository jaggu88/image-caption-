IMAGE CAPTION GENERATOR

Introduction: 
Being able to automatically describe the content of an image using properly formed English sentences is a very challenging task, but it could have great impact, for instance by helping visually impaired people better understand the content of images on the web. Image captioning aims to automatically generate a sentence description for an image. It takes an image as input and generates an English sentence as output, describing the contents of the image. The task is rather complex, as the concepts of both computer vision and natural language processing domains are combined. The working model would be developed using the concepts of a Convolution Neural Network (CNN) and Long Short Term Memory (LSTM) model. The CNN works as an encoder to extract features from images and LSTM works as a decoder to generate words describing image. This problem is multimodal, which generates the need to construct a hybrid model that can leverage the problem's multidimensionality. Approaches such as prototype based, and retrieval based  approaches  have  historically  been  used  to  solve  the  issue.
Problem statement:
To develop a system for users, which can automatically  generate the description of an image with the use of CNN along with LSTM. Automatically describing the content of images using natural language is a fundamental and challenging task. With the advancement in computing power along with the availability of huge datasets, building models that can generate captions for an image has become possible. Given a picture, it is natural for a person to explain an immense number of details about this image with a fast glance. Although great development has been made in computer vision, tasks such as recognizing an object, action classification, image classification, attribute classification and scene recognition are possible, but it is a relatively new task to let a computer describe an image that is forwarded to it in the form of a human-like sentence.
Flow chart and steps involved

 

Step 1: Import the required libraries
Dataset 
•	Flicker8k_Dataset – Dataset folder which contains 8091 images.
•	Flickr_8k_text – Dataset folder which contains text files and captions of images.

Step 2: Load images dataset which contains list of names of all the images
Step 3: Extracting the feature vector 
Step 4: Pre-process the images (transfer all the images to our predefined model (used for feature extraction) )
Step 5: Load the descriptions
The format of our file is image and caption separated by a newline (“\n”) i.e., it consists of the name of the image followed by a space and the description of the image in CSV format. so, we need to map the image to its descriptions by storing them in a dictionary. 
Step 6: Pre-process and cleaning the text 
•Splitting the image name and caption and then storing it in a dictionary(image name as key and caption as value)
•Adding <start> and <end> identifier for each caption. We need this so that our LSTM model can identify the starting and ending of the caption.
•One of the main steps in NLP is to remove noise so that the machine can detect the patterns easily in the text. Noise will be present in the form of special characters such as hashtags, punctuation, converting all text to lowercase. All of which are difficult for computers to understand if they are present in the text. So, we need to remove these for better results.
•Creating a vocabulary which will contain all  the words we have in our captions
•Converting string(captions) to integers and storing in dictionary
Step 7: Create data generator 
The input to our model is [x1, x2] and the output will be y, where x1 is the  feature vector of that image, x2 is the input text sequence and y is the output text sequence that the model must predict.
e.g., cat sitting on a table 
1. <start> cat sitting on a table <end>
2. <start>   -- cat    { so we will take  <start> (x1)and feature vector(x2) to predict out next word i.e., cat (y)
3. <start> cat-- sitting
Step 8: Defining CNN-RNN model 
•Feature Extractor – The feature extracted from the image has a size of 2048, with a dense layer, we will reduce the dimensions to 256 nodes.
•Sequence Processor – An embedding layer will handle the textual input, followed by the LSTM layer.
•Decoder – By merging the output from the above two layers, we will process by the dense layer to make the final prediction. The final layer will contain the number of nodes equal to our vocabulary size.
Step 9: Training our model 
Step 10 :Testing our model 
Algorithms 
1) Convolutional Neural Network 
Convolutional Neural networks are specialized deep neural networks which processes the data that has input shape like a 2D matrix. CNN works well with images and are easily represented as a 2D matrix. Image classification and identification can be easily done using CNN. It can determine whether an image is a bird, a plane or Superman, etc. Important features of an image can be extracted by scanning the image from left to right and top to bottom and finally the features are combined to classify images. It can deal with the images that have been translated, rotated, scaled and changes in perspective.
2) Long Short Term Memory LSTM 
LSTM are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. We can predict what the next words will be based on the previous text. It has shown itself effective from the traditional RNN by overcoming the limitations of RNN. LSTM can carry out relevant information throughout the processing, it discards non-relevant information.
Author	Pre-processing and feature extraction	Feature used for classification	Model	Dataset	Results(Accuracy of classification in %)
Grishma Sharma
2019	1) VGG16 Architecture(extract features from image)

2)Pre-processing the caption		CNN(VGG), LSTM and GRU	Flickr 8k	
Dr. Vinayak D.Shinde, Mahiman P. Dave
2020	Feature extraction using CNN		CNN and LSTM	Flickr 8k	
Oriol Vinyals,Alexander Toshev 
2015	Feature extraction using CNN		CNN and LSTM	Pascal,Flickr30k,COCO and SBU	Bleu Score of all the datasets:-
Pascal-0.59
Flickr30k-0.66
COCO-0.277
SBU-0.28
Dinesh Sreekanthan,Amutha A.L
2018	VGG16 Architecture		CNN and LSTM	Flickr 8k	Bleu Score-0.683
B.Krishnakumar,K.Kousalya
2020	VGG16 Architecture		CNN and LSTM	MSCOCO	
Jiuxiang Gu , Gang Wang, Jianfei Cai
	Feature extraction using CNN		CNN,LSTM and GRU	MSCOCO and Flickr 30k	Bleu Score of all the datasets:-
Flickr 30k-0.57
MSCOCO-0.62
Qichen Fu , Yige Liu, Zijian Xie	1)Preprocessing of image

2)Preprocessing of captions

3)Feature extraction using CNN		CNN and LSTM	Flickr 8k	
Yongqing Zhu, Xiangyang Li, Xue Li, Jian Sun	VGGNet		CNN and LSTM	MSCOCO	Bleu Score of 0.16
Raimonda Staniute and Dmitrij Šešok 
2019	ResNet-50		CNN and LSTM	MSCOCO	Bleu Score of 0.75
 Prachi Waghmare and Dr. Swati Shinde
	VGG16 		CNN and LSTM	Flickr8k 	Bleu score 
Moses Soh	CNN		CNN and LSTM 	MSCOCO	Bleu score :24.4 
METEOR : 21.5 
CIDER :81.7
Simao Herdade, Armin Kappeler, Kofi Boakye, Joao Soares	CNN 		CNN and LSTM 	MSCOCO	BLEU:80.5
CIDEr-D:
128.3
SPICE:22.6
METEOR:
28.7
ROUGE-L:
58.4
Christopher Elamri, Teun de Planque	VGG16
,PCA(dimensionality reduction)		CNN and LSTM 	MSCOCO	BLEU:62.5 
METEO:19.4
CIDEr: 65.8

Literature review:
Dr. Vinayak D. Shinde, Mahiman P. Dave used Flickr_8K dataset. The model used deep neural networks such as CNN which processes the data that has input shape like a 2D matrix and LSTM which is used in RNN. CNN model is used  to extract features of an image. These features are then fed into a LSTM model to generate a description of the image in grammatically correct English sentences describing the surroundings. Oriol Vinyals, Alexander Toshev used feature extraction using CNN on dataset Pascal,Flickr30k,COCO and SBU which made an accuracy of 0.59-Pascal, 0.66- Flickr30k, 0.277- COCO, 0.28- SBU. Garima Sharma used data set Flickr8k with CNN VGG architecture, LSTM and GRU. Dinesh Sreekanthan, Amutha A.L used dataset Flickr8k with CNN-VGG and LSTM which made an accuracy of Bleu Score- 0.683. B.Krishnakumar, K.Kousalya used CNN-VGG and LSTM on MSCOCO dataset. Yongqing Zhu, Xiangyang Li, Xue Li, Jian Sun used CNN-VGG.Net and LSTM on MSCOCO with an accuracy of 0.16. B.Krishnakumar, K.Kousalya
Used MSCOCO dataset which includes 8091 images of 500X333pixels and has to be pre-defined dataset of 6000images, developed dataset of 1000 images and test dataset of 1000 images. The model used were VGG16-CNN . Raimonda Staniute and Dmitrij Šešok used dataset on MSCOCO with CNN encoder and RNN decoder. The model generated words to represent the image with a full grammatical and stylistically correct sentence. ResNet-50 is used for feature selection and LSTM was used for sequential work which made an accuracy of bleu score of 0.75. 
