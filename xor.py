import tensorflow as tf
sess = tf.InteractiveSession()

TRAINING_INPUTS = [[0,0], [0,1], [1,0], [1,1]]
TRAINING_OUTPUTS = [[1,0], [0,1], [0,1], [1,0]]
'''
can't make it work with outputs which are different "shapes"
pretend [[1,0],[0,1],[0,1],[1,0]] (shape [4,2]) == [[0],[1],[1],[0]] (shape[4,1])
'''
NO_OF_HIDDEN_NEURONS = 25

inputPlaceHolder = tf.placeholder("float", [4,2])
outputPlaceHolder = tf.placeholder("float", [4,2]) 
#[4,2] because [[0,0], [0,0], [0,0], [0,0]] (4 elements of 2)

#for all comments when I say 25 it's because i'm assuming 25 is NO_OF_HIDDEN_NEURONS, it may change
#This gets a shape of [2,25] (if 25 is NO_OF_HIDDEN_NEURONS) randomly init between -.01, .01
firstLayerWeights = tf.Variable(tf.random_uniform([2, NO_OF_HIDDEN_NEURONS], -.01, .01))
#This gets a shape of [?,25], so just an array of 25 numbers between -.01 and .01
firstLayerBias = tf.Variable(tf.random_uniform([NO_OF_HIDDEN_NEURONS], -.01, .01))
#tf.matmul = maths multiply
#This is the rectified linear activation function.
#Basically a neuron on the first layer will fire if the below is true for it (I think!)
firstLayerOutput = tf.nn.relu(tf.matmul(inputPlaceHolder,firstLayerWeights) + firstLayerBias)

secondLayerWeights = tf.Variable(tf.random_uniform([NO_OF_HIDDEN_NEURONS, 2], -.1, .1))
secondLayerOutput = tf.matmul(firstLayerOutput, secondLayerWeights)
'''
this is the activation function for the second layer 
https://en.wikipedia.org/wiki/Softmax_function
My current understanding of softmax is that is increases the weights for the 
largest numbers and squashes the rest
'''
output = tf.nn.softmax(secondLayerOutput)

''' 
Cross entropy is basically training a neuron on multiple inputs
see below for more details:
http://neuralnetworksanddeeplearning.com/chap3.html
It has a minus in front because the crossentropy functions do 
I'm not 100% on the maths yet...

'''
crossEntropy = -tf.reduce_sum(outputPlaceHolder*tf.log(output))
trainStep = tf.train.GradientDescentOptimizer(0.2).minimize(crossEntropy)

tf.initialize_all_variables().run()
for step in range(1000):
	feedDictionary = {inputPlaceHolder: TRAINING_INPUTS, outputPlaceHolder:TRAINING_OUTPUTS}
	entropy,_=sess.run([crossEntropy,trainStep],feedDictionary)
	if entropy<1:break 
	print("step %d : entropy %s" % (step,entropy))
	
#https://www.tensorflow.org/versions/r0.11/api_docs/python/math_ops.html#argmax
correctPrediction = tf.equal(tf.argmax(output,1), tf.argmax(outputPlaceHolder,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float")) # [True, False, True, True] -> [1,0,1,1] -> 0.75.. We want 100% ideally

print("accuracy %s"%(accuracy.eval({inputPlaceHolder: TRAINING_INPUTS, outputPlaceHolder: TRAINING_OUTPUTS})))

learnedOutput=tf.argmax(output,1)
print(learnedOutput.eval({inputPlaceHolder: TRAINING_INPUTS}))
#print(tf.random_uniform([2, NO_OF_HIDDEN_NEURONS], -.01, .01).eval())
#print(tf.random_uniform([NO_OF_HIDDEN_NEURONS], -.01, .01).eval())
#print(tf.random_uniform([NO_OF_HIDDEN_NEURONS, 2], -.01, .01).eval())
