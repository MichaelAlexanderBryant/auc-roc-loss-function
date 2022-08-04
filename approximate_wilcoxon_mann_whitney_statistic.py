import tensorflow as tf

def ApproximateWilcoxonMannWhitneyStatistic(y_true, y_pred):
    
    """Computes an approximation of the normalized Wilcoxon-Mann-Whitney (WMW)
    statistic using labels and predictions.
    
    The normalized WMW statistic can be shown to be equal to the AUC-ROC. However,
    it is a step function so it is not differentiable. The normalized WCW
    statistic can be approximated with a smooth, differentiable function
    which makes the approximated version a near ideal loss function for optimizing
    for the AUC-ROC metric.
    
    The loss function has two parameters, gamma and p, which are recommended
    to be kept between 0.1 to 0.7 and at 2 or 3, respectively.
    
    For more information:
    Optimizing Classifier Performance via an Approximation to the
    Wilcoxon-Mann-Whitney Statistic. Yan, Lian and Dodier, Robert H. and Mozer,
    Michael and Wolniewicz, Richard H. International Conference on Machine
    Learning (2003).
    """ 
    
    # Define parameters.
    GAMMA = .7
    P = 2
    
    # Convert labels and predictions to float64.
    y_true = tf.cast(y_true, dtype="float64")
    y_pred = tf.cast(y_pred, dtype="float64")
    
    # Boolean vector for determining positive and negative labels.
    boolean_vector = tf.greater_equal(y_true, 0.5)
    
    # Mask predictions to seperate true positive and negatives.
    positive_predictions = tf.boolean_mask(y_pred, boolean_vector)
    negative_predictions = tf.boolean_mask(y_pred, ~boolean_vector)
    
    # Obtain size of positive and negative prediction arrays.
    M = tf.reduce_sum(tf.cast(boolean_vector, dtype="float64"))
    N = tf.reduce_sum(tf.cast(~boolean_vector, dtype="float64"))
    
    # Reshape arrays into original shape.
    positive_predictions = tf.reshape(positive_predictions, shape=(M,1))
    negative_predictions = tf.reshape(negative_predictions, shape=(N,1))

    # Convert gamma parameter to float64.
    gamma_array = tf.constant(GAMMA, dtype="float64")
    
    # Broadcast gamma parameter to MxN matrix.
    gamma_array = tf.broadcast_to(gamma_array, shape=(M,N))
    
    # Broadcast positive predictions to MxN.
    positive_predictions = tf.broadcast_to(positive_predictions, shape=(M,N))
    
    # Broadcast negative predictions to NxM then transpose.
    negative_predictions = tf.transpose(tf.broadcast_to(negative_predictions,
                                               shape=(N,M)))
    
    # Subtract positive predictions matrix from negative predictions matrix.
    sub_neg_pos = tf.subtract(negative_predictions, positive_predictions)
    
    # Add gamma matrix to subtracted negative/positive matrix.
    add_gamma = tf.add(sub_neg_pos, gamma_array)
    
    # Check if positive predictions are less than negative predictions plus
    # gamma.
    inequality_check = tf.math.less(tf.subtract(positive_predictions,
                                                negative_predictions), GAMMA)
    
    # Convert Boolean values to float64.
    inequality_check = tf.cast(inequality_check, dtype="float64")
    
    # Element-wise multiplication which effectively masks values that do not
    # meet inequality criterion.
    inequality_mask = tf.math.multiply(inequality_check, add_gamma)
    
    # Element-wise raise to power P.
    raise_to_p = tf.math.pow(inequality_mask, P)
    
    # Sum all elements.
    loss = tf.reduce_sum(raise_to_p)
    
    return loss
