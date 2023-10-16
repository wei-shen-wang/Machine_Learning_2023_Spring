# Machine Learning 2023 Spring
These are the assignment of Machine Learning course instructed by Prof. Hung-Yi Lee.

## HW01 Regression
* **Task**

  Predict the COVID-19 tested positive cases with basic DNN regression network.
* **My adjustment for performance**

  Select features with higher correlation to tested positive cases. Choose better optimizer, usually Adam or its variant.

## HW02 Classification
* **Task**

  Phoneme Classification with DNN network for classification.
* **My adjustment for performance**

  Add learning rate scheduler[2], batch normalization and dropout.
  Learning rate scheduler for better optimal point. Batch normalization for faster convergence. Dropout to avoid overfitting.
  According to observation on the training data, the scenario where one particular phoneme is in a class that is different to both the phoneme right before and right after it should not happen. Therefore, after the model's prediction, I post process the inference data by changing that particular phoneme to eliminate the anomaly.
  ```
    original inference data
    16 17 16
    post process
    16 16 16 
  ```


## HW03 CNN
* **Task**

  Learn CNN with image classification.

* **My adjustment for performance**

  Use the predefined model ResNeXt[3], and choose the optimizer (*Adam is not always the best*) and learning rate schedule after reading the paper of ResNet[4]. The reason that ResNet has such great performance on image classification is due to its deep residual network architecture. Traditional DNN's performance will eventually degrades if there are too many layers in the model architecture. ResNet avoid the downgrade of the performance by directly adding outputs of earlier layer to latter layer skipping several layers in between(e.g. the output of layer 1 is directly added to layer 5). This kind of networks stabilize the network and make sure the network improve gradually and carefully. In addition, I implemented data augmentation by calling the api[5] in torchvision. Finally, normalization of the whole image is implemented for faster convergence.

## HW04 Self-Attention
* **Task**

  Speaker Identification with Self-Attention.

* **My adjustment for performance**

  Use the Transformer encoder and add convolution layer after each encoder layer.

## HW05 Transformer
* **Task**

  Implement machine translation with the Transformer architecture.

* **My adjustment for performance**

  After reading the paper Attention Is All You Need[6], tune the hyperparameter according to the paper. Larger model usually result in better performance but requires much more time and resources which I don't have for an one week homework. Therefore, I chose to implement model based on the base transformer model in the paper.

## HW06 Generative Model
* **Task**

  Anime face generation using Generative Model.

* **My adjustment for performance**

  Try different pre-defined model, such as styleGAN2 and denoising diffusion model.
  Ideally speaking, denoising diffusion model should generate anime face with better score than GAN model, but due to the limitation of time and resources. My trial of GAN achieved better performance then my trial of denoising diffusion model. Although denoising diffusion model is the current state of art image generative model, but it seems that much more time is required to train it in order to achieve higher quality.

## HW07 BERT
* **Task**

  Fine tune pretrained BERT[7] model on huggingface[8] to do extractive question answering.

* **My adjustment for performance**

  Add randomness when picking the range of doc stride and the answer location in it as the answer is not always in the middle of the doc stride. Sometimes the model output's end_index is larger than start_index, neglect this case to avoid wrong output.

## HW08 Auto-encoder
* **Task**

  Use auto-encoder to do human face image anomaly detection. Images with anomaly should not be able to be reconstructed by the auto-encoder where as the normal images can.

* **My adjustment for performance**

  Adjust the bottleneck size of the auto-encoder to a certain number so that the information of a normal image can be encoded. The bottleneck should not be too small as the auto-encoder will not be able to reconstruct the image no matter if it is from the normal distribution or not. The bottleneck also should not be too large because the image will easily be reconstructed(simply duplicated) and the anomaly will not be able to be detected. In addition, I did center crop to keep out useless information like the background behind the human face. I made this decision because after reforming the images from the `.npy` file and discovering that almost every images's background is very different to one another. This method resulted in better performance because most of the backgrounds' varieties are directly discarded.

## HW09 Explainable AI
* **Task**

  Answer the questions after running the code provided by TAs. The code's objective is aimed to make the model architecture and its result of HW3 and HW7 explainable.

* **My adjustment**

  Take a look at some of the `lime` documentations. Learn technique for explaining image classification models, including saliency map, smooth grad ,filter activation and integrated gradients. Implement cosine_similarity and euclidean distance for comparison of BERT word embeddings.

## HW10 Attack
* **Task**

  Attack a black box image classification model while possessing the the training data the model. Train a proxy model to do so.

* **My adjustment for performance**

  Implement the MIFGSM[9] for better performance. Ensemble pretrained multiple models with different architecture and number of layers to avoid overfitting to successful attack on specific model.


Reference

[1] https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php

[2] https://pytorch.org/docs/stable/optim.html

[3] https://arxiv.org/abs/1512.03385

[4] https://arxiv.org/abs/1611.05431

[5] https://pytorch.org/vision/stable/transforms.html

[6] https://arxiv.org/abs/1706.03762

[7] https://arxiv.org/abs/1810.04805

[8] https://huggingface.co/

[9] https://arxiv.org/abs/1710.06081
