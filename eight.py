from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=True)
print("Training data size:" ,mnist.train.num_examples)
print("Validating data size:" ,mnist.validation.num_examples)
print("testing data size:" ,mnist.test.num_examples)
print("example training data:", mnist.train.images[0])
print("example training data label" ,mnist.train.labels[0])
batch_size = 100
xs ,ys = mnist.train.next_batch(batch_size)
print("x shape" ,xs.shape)
print("y shape" ,ys.shape)