# Machine Learning in Applications Project : Detection of Anomalous Behaviour in Industrial Robot


In this project, the capabilities of autoencoders on time-series anomaly detection (TAD) based on reconstruction cost will be studied with all its details.

Objectives:
* Implement an autoencoder model that has capacity to be able to generate same output with input. (Capacity: to have enough trainable parameters to handle reconstruction of time-series.)
* Demonstrate the importance of evaluation method selection in TAD. 
* Compare traditional training techniques by adversarial training with adversarial autoencoder. Same autoencoder will be used  to demonstrate regularization phase of adversarial learning better.

## Autoencoder

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/AE.png" alt="drawing" width="600" height="250"/>

An autoencoder is a type of artificial neural network used to learn efficient codings of unlabelled data. A 5 layer LSTM autoencoder has been selected with latent size 10 and 476,199 trainable parameters. Single LSTM autoencoder is suggested in reference paper [1]. However, more complex model has been chosen because of data complexity. (The learnable parameter would be much more than needed. However, optimizing number of learnable parameters is not a objective of this project.)



## Evaluation Method

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


## Adversarial Autoencoder

"The Adversarial Autoencoder (AAE) is a brilliant concept that combines the autoencoder architecture with GAN's adversarial loss notion. It works similarly to the Variational Autoencoder (VAE), except instead of KL-divergence, it utilizes adversarial loss to regularize the latent code.
"

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/AAE.png" alt="drawing" width="600" height="250"/>

The same autoencoder has been selected to demonstrate better the differences made by adversarial learning. The generator is the encoder of the autoencoder. A fully connected model has been selected with 18,049 trainable-parameters as the discriminator.

## Results

### Random initialized model vs trained model

**Best F1 Scores with different K values:**

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/best.png" alt="drawing" width="600" height="200"/>

**Mean F1 Scores with different K values:**

<img src="https://github.com/shadow036/machine-learning-in-applications/blob/can/img/mean.png" alt="drawing" width="600" height="200"/>

to do: write what you see in graphs 

### Traditional vs adversarial learning

to do

## References
1) Kim, S., Choi, K., Choi, H. S., Lee, B., & Yoon, S. (2022, June). Towards a rigorous evaluation of time-series anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 7, pp. 7194-7201).
2) Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
3) Bank, D., Koenigstein, N., & Giryes, R. (2020). Autoencoders. arXiv preprint arXiv:2003.05991.