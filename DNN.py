# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import tensorflow as tf
import time


# 256 neurons in each hidden layers
n_hidden_1 = 401
n_hidden_2 = 300
n_hidden_3 = 200
n_hidden_4 = 100
n_hidden_5 = 50

# input size is the size of a picture: 28*28
# output size
input_size = 440
output_size = 3


# Parameters
learning_rate = 0.001
training_epochs = 2500
batch_size = 32
display_step = 1



def layer1(x, weight_shape, bias_shape):
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - weight_shape: shape the the weight maxtrix
        - bias_shape: shape of the bias vector
    output:
        - output vector of the layer after the matrix multiplication and transformation
    """
    
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    
    return tf.nn.softmax(tf.matmul(x, W) + b)


def layer2(x, weight_shape, bias_shape):
    """
    Defines the network layers
    input:
        - x: input vector of the layer
        - weight_shape: shape the the weight maxtrix
        - bias_shape: shape of the bias vector
    output:
        - output vector of the layer after the matrix multiplication and transformation
    """
    
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    
    return tf.nn.relu(tf.matmul(x, W) + b)


def inference(x):
    """
    define the whole network (5 hidden layers + output layers)
    input:
        - a batch of pictures 
        (input shape = (batch_size*image_size))
    output:
        - a batch vector corresponding to the logits predicted by the network
        (output shape = (batch_size*output_size)) 
    """
    
    with tf.variable_scope("hidden_layer_1"):
        hidden_1 = layer2(x, [input_size, n_hidden_1], [n_hidden_1])
        #print([input_size, n_hidden_1])
     
    with tf.variable_scope("hidden_layer_2"):
        hidden_2 = layer2(hidden_1, [n_hidden_1, n_hidden_2], [n_hidden_2])
        #print([n_hidden_1, n_hidden_2])
        
    with tf.variable_scope("hidden_layer_3"):
        hidden_3 = layer2(hidden_2, [n_hidden_2, n_hidden_3], [n_hidden_3])
        #print([n_hidden_2, n_hidden_3])
        
    with tf.variable_scope("hidden_layer_4"):
        hidden_4 = layer2(hidden_3, [n_hidden_3, n_hidden_4], [n_hidden_4])
        #print([n_hidden_3, n_hidden_4])
        
    with tf.variable_scope("hidden_layer_5"):
        hidden_5 = layer2(hidden_4, [n_hidden_4, n_hidden_5], [n_hidden_5])
        #print([n_hidden_4, n_hidden_5])
     
    with tf.variable_scope("output"):
        output = layer1(hidden_5, [n_hidden_5, output_size], [output_size])
        #print([n_hidden_5, output_size])

    return output


def loss(output, y):
    """
    Computes softmax cross entropy between logits and labels and then the loss 
    
    intput:
        - output: the output of the inference function 
        - y: true value of the sample batch
        
        the two have the same shape (batch_size * num_of_classes)
    output:
        - loss: loss of the corresponding batch (scalar tensor)
    
    """
    #Computes softmax cross entropy between logits and labels.
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
    loss = tf.reduce_mean(xentropy)

    return loss

def training(cost, global_step):
    """
    defines the necessary elements to train the network
    
    intput:
        - cost: the cost is the loss of the corresponding batch
        - global_step: number of batch seen so far, it is incremented by one each time the .minimize() function is called
    """
    tf.summary.scalar("cost", cost)
    # using Adam Optimizer 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def evaluate(output, y):
    """
    evaluates the accuracy on the validation set 
    input:
        -output: prediction vector of the network for the validation set
        -y: true value for the validation set
    output:
        - accuracy: accuracy on the validation set (scalar between 0 and 1)
    """
    #correct prediction is a binary vector which equals one when the output and y match
    #otherwise the vector equals 0
    #tf.cast: change the type of a tensor into another one
    #then, by taking the mean of the tensor, we directly have the average score, so the accuracy
    
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation_error", (1.0 - accuracy))
    return accuracy



if __name__ == '__main__':
    
    #please, make sure you changed for your own path 
    log_files_path = 'C:/Users/Othman/logs/CNNs/'
    start_time = time.time()
    
    with tf.Graph().as_default():

        with tf.variable_scope("MNIST_convoultional_model"):
            #neural network definition
            
            #the input variables are first define as placeholder 
            # a placeholder is a variable/data which will be assigned later 
            # MNIST data image of shape 28*28=784
            x = tf.placeholder("float", [None, 784]) 
            # 0-9 digits recognition
            y = tf.placeholder("float", [None, 10])  
            
            # dropout probability
            keep_prob = tf.placeholder(tf.float32) 
            #the network is defined using the inference function defined above in the code
            output = inference(x, keep_prob)
            cost = loss(output, y)
            #initialize the value of the global_step variable 
            # recall: it is incremented by one each time the .minimise() is called
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = training(cost, global_step)
            #evaluate the accuracy of the network (done on a validation set)
            eval_op = evaluate(output, y)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session()
            
            summary_writer = tf.summary.FileWriter(log_files_path, sess.graph)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            # Training cycle
            for epoch in range(training_epochs):

                avg_cost = 0.0
                total_batch = int(mnist.train.num_examples/batch_size)
                
                # Loop over all batches
                for i in range(total_batch):
                    
                    minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                    
                    # Fit training using batch data
                    sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})
                    
                    # Compute average loss
                    avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.5})/total_batch
                
                
                # Display logs per epoch step
                if epoch % display_step == 0:
                    
                    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:0.9f}".format(avg_cost))
                    
                    #probability dropout of 1 during validation
                    accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1})
                    print("Validation Error:", (1 - accuracy))
                    
                    # probability dropout of 0.25 during training
                    summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: 0.25})
                    summary_writer.add_summary(summary_str, sess.run(global_step))
                    
                    saver.save(sess, log_files_path+'model-checkpoint', global_step=global_step)
                    
            print("Optimization Done")
                    
            accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})
            print("Test Accuracy:", accuracy)
                    
        elapsed_time = time.time() - start_time
        print('Execution time was %0.3f' % elapsed_time)



