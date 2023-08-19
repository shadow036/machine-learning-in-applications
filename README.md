# Machine Learning in Applications Project : Detection of Anomalous Behaviour in Industrial Robot


In this project, the capabilities of autoencoders on time-series anomaly detection (TAD) based on reconstruction cost will be studied with all its details.

Objectives:
* Implement an autoencoder model that has capacity to be able to generate same output with input. (Capacity: to have enough trainable parameters to handle reconstruction of time-series.)
* Demonstrate the importance of evaluation method selection in TAD. 
* Compare traditional training techniques by adversarial training with adversarial autoencoder. Same autoencoder will be used  to demonstrate regularization phase of adversarial learning better.

## Data

**Data Normal:** The data that collected during production with normal velocity.

**Data Slow:** The data that collected during production with slow velocity.

Both normal and slow data has 86 features that from action to sensor values. Permanent anomalies are the anomalies effects production velocity (just in data slow). Temporal anomalies are the anomalies by reading abnormal values from sensor (in both data).

**A selected feature from data:**

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/data.png" alt="drawing" width="600" height="200"/>

(Left: data normal, Right: data slow)

As it can be observable, there are more temporal anomalies in data normal than data slow.

## Autoencoder [3]

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/AE.png" alt="drawing" width="600" height="250"/>

An autoencoder is a type of artificial neural network used to learn efficient codings of unlabelled data. A 5 layer LSTM autoencoder has been selected with latent size 10 and 476,199 trainable parameters. Single LSTM autoencoder is suggested in reference paper [1]. However, more complex model has been chosen because of data complexity. (The learnable parameter would be much more than needed. However, optimizing number of learnable parameters is not a objective of this project.)



## Evaluation Method [1]

Experiment about evaluation methods on TAD in reference paper [1] is repeated in this section.

**Point Adjustment (PA):** 

"If at least one moment in a contiguous anomaly segment is detected as an anomaly, the entire segment is then considered to be correctly predicted as an anomaly. 

Most of the Time-Series Anomaly Detection (TAD) methods measure the F1 score after applying this peculiar evaluation protocol."

The allegation: PA greatly overestimates the detection performance.

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/EM1.png" alt="drawing" width="400" height="500"/>

**Proposed solution PA%K:**

"Apply PA to the set only if the ratio of the number of correctly detected anomalies in the set to its length exceeds the PA%K threshold, K. 

Mitigate the overestimation effect of F1PA & the possibility of underestimation of F1.

K can be selected manually between 0 and 100 based on prior information. (If test labels are reliable, higher K. And vice versa.)"

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/EM2.png" alt="drawing" width="600" height="250"/>


## Adversarial Autoencoder [2]

"The Adversarial Autoencoder (AAE) is a brilliant concept that combines the autoencoder architecture with GAN's adversarial loss notion. It works similarly to the Variational Autoencoder (VAE), except instead of KL-divergence, it utilizes adversarial loss to regularize the latent code.
"

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/AAE.png" alt="drawing" width="600" height="250"/>

The same autoencoder has been selected to demonstrate better the differences made by adversarial learning. The generator is the encoder of the autoencoder. A fully connected model has been selected with 18,049 trainable-parameters as the discriminator.


### Semi-Supervised Autoencoder

In semi-supervised AAE, "There are two separate adversarial networks that regularize the hidden representation of the autoencoder. The first adversarial network imposes a Categorical distribution on the label representation. This adversarial network ensures that the latent class variable y does not carry any style information and that the aggregated posterior distribution of y matches the Categorical distribution. The second adversarial network imposes a Gaussian distribution on the style representation which ensures the latent variable z is a continuous Gaussian variable.".

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/aae_semi.png" alt="drawing" width="400" height="400"/>

In this experiment, only the first adversarial network will be used to have leaning to construct more repetitive samples than scare occurred samples which may be temporal anomalies.

**Training details of adversarial autoencoder:**

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/aae loss.png" alt="drawing" width="300" height="300"/>


## Results

### Random initialized model vs trained model

**Best F1 Scores with different K values:**

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/best.png" alt="drawing" width="600" height="200"/>

**Mean F1 Scores with different K values:**

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/mean.png" alt="drawing" width="600" height="200"/>

(Left: trained model, Right: untrained model)

As expected, the PA approach apt to make overestimated evaluation. Best f1-score reaches higher than 0.9 with low K values for untrained model. 

Additionally, it would be worth to mention that best f1-score can be higher than 0.65 for untrained model, even without PA. And best f1-scores are 1 with lower K values than 50 for trained model. Because of that mean f1-score would be better option to make more general and/or honest evaluations.

About K value, the reference paper [1] mentions that K value must be picked manually with consideration of label reliability. K would be between 30 and 50 for mean f1-score evaluation, 60-80 for best f1-score in this case.

### Adversarial Learning


| Best F1-Scores                                                                                                                               |      Mean F1-Scores |
|----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/aae_best.png" alt="drawing" width="300" height="200"/>  |  <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/aae_mean.png" alt="drawing" width="300" height="200"/>|

AAE reached very similar results in comparison with AE.

Note: Both AAE & AE models have been trained for 50 epochs.

## Discussion & Further Improvements

### Temporal Anomalies

Autoencoder models often become able to well reconstruct also the anomalies in the data. This phenomenon is more evident when there are anomalies in the training set. In particular when these anomalies are labeled, a setting called semi-supervised, the best way to train autoencoders is to ignore anomalies and minimize the reconstruction error on normal data. And model AE-SAD offers solution for this issue according to it. [4]

As we know, the normal data includes many temporal anomalies (abnormal readings). Therefore, the performance of the model will have been increased, if temporal anomalies could be avoided in training. Because of that, the loss function will be changed as following to avoid or learn less from temporal anomalies. 

| Before (Mean Squared Error)                                                                                                           | After                                                                                                                                 |
|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/eq1.png" alt="drawing" width="150" height="50"/> | <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/eq2.png" alt="drawing" width="300" height="50"/> |

However, a preprocessing step is needed to determine temporal anomalies. Therefore, most radical 1% of readings will be labelled as temporal anomaly for each feature. And if any of the feature value is in the 1%, the sample will be labelled as anomaly. 

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/threshold anomalies.png" alt="drawing" width="300" height="300"/>

At the end, estimated 5% of the samples are labelled as temporal anomaly.


**Training details of AE-SAD model**

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/SAD loss.png" alt="drawing" width="300" height="300"/>


### Results of Improvement


| Baseline - Mean F1-Scores                                                                                                                               | AE-SAD - Mean F1-Scores                                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/result baseline mean.png" alt="drawing" width="300" height="300"/> | <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/result SAD mean.png" alt="drawing" width="300" height="300"/> |

As it can be observable, AE-SAD model could perform better than the baseline model. Which means, it can be said that abnormal readings must be taken care of in the training phase.


### Mean vs Best F1-Score for Evaluation

An evaluation method should show the difference of performances clearly. However, best F1-Score with point adjustment (PA) would be bad choice for apparency of difference in many K. 


| Baseline - Best F1-Scores                                                                                                                               | AE-SAD - Best F1-Scores                                                                                                                            |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/result baseline best.png" alt="drawing" width="300" height="300"/> | <img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/result SAD best.png" alt="drawing" width="300" height="300"/> |

As it is obvious, for many K values, especially below 50, it is hard to make an evaluation with PA between models. 


## References
1) Kim, S., Choi, K., Choi, H. S., Lee, B., & Yoon, S. (2022, June). Towards a rigorous evaluation of time-series anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 7, pp. 7194-7201).
2) Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
3) Bank, D., Koenigstein, N., & Giryes, R. (2020). Autoencoders. arXiv preprint arXiv:2003.05991.
4) Angiulli, F., Fassetti, F., & Ferragina, L. (2023). Reconstruction Error-based Anomaly Detection with Few Outlying Examples. arXiv preprint arXiv:2305.10464.