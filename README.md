# auc-roc-loss-function

A TensorFlow loss function for binary classification based on an approximation of the normalized Wilcoxon-Mann-Whitney (WMW) statistic.
 
The normalized WMW statistic can be shown to be equal to the AUC-ROC. However, it is a step function so it is not differentiable. The normalized WCW statistic can be approximated with a smooth, differentiable function which makes the approximated version a near ideal loss function for optimizing for the AUC-ROC metric.
    
The loss function has two parameters, gamma and p, which are recommended to be kept between 0.1 to 0.7 and at 2 or 3, respectively.
    
For more information:
Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic. Yan, Lian and Dodier, Robert H. and Mozer, Michael and Wolniewicz, Richard H. International Conference on Machine Learning (2003). [Link](https://www.aaai.org/Papers/ICML/2003/ICML03-110.pdf)

[Click here to see an example using this loss function.](https://github.com/MichaelABryant/tps-aug-2022/blob/main/logistic-regression-with-custom-loss-function.ipynb)
