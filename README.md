# auc-roc-loss-function

A TensorFlow loss function that computes an approximation of the normalized Wilcoxon-Mann-Whitney (WMW) statistic between labels and predictions.
 
The normalized WMW statistic can be shown to be equal the AUC-ROC. However, it is a step function so it is not differentiable. The normalized WCW statistic can be approximated with a smooth, differentiable function which makes the approximated version an ideal loss function for optimizing the AUC-ROC metric.
    
The loss function has two parameters, gamma and p, which are recommended to be kept between 0.1 to 0.7 and at 2 or 3, respectively.

>     model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics= METRIC,
            loss=LOSS)
    
For more information:
Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic. Yan, Lian and Dodier, Robert H. and Mozer, Michael and Wolniewicz, Richard H. International Conference on Machine Learning (2003). [Link](https://www.aaai.org/Papers/ICML/2003/ICML03-110.pdf)
