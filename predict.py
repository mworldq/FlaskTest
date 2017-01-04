#import modules
import sys
import tensorflow as tf
from PIL import Image,ImageFilter

def predictint(imvalue):
    # Initializing the variables
    init = tf.initialize_all_variables()

    ########################## Copy From create_model.py ##########################################
    # Parameters
    learning_rate = 0.01
    training_epochs = 50
    batch_size = 100
    display_step = 5
    model_path = "./model-logistic-share.ckpt"

    # Create the model
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

    # Set model weights
    def get_scope_variable(scope_name, var, shape=None):
        with tf.variable_scope(scope_name) as scope:
            try:
                v = tf.get_variable(var, shape)
            except ValueError:
                scope.reuse_variables()
                v = tf.get_variable(var)
        return v

    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    W = get_scope_variable("conv1", "W", [784, 10])
    b = get_scope_variable("conv1", "b", [10])

    # Construct model
    pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    saver = tf.train.Saver()
    ####################################################################
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)
        # print ("Model restored.")

        prediction = tf.argmax(pred, 1)
        return prediction.eval(feed_dict={x: [imvalue]}, session=sess)



#Copy from predict_1.py
def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheigth = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheigth == 0):  # rare case but minimum is 1 pixel
            nheigth = 1
            # resize and sharpen
        img = im.resize((20, nheigth), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheigth) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
    # print(tva)


"""
Main function.
"""
if __name__ == "__main__":
    imvalue = imageprepare(sys.argv[1])
    predint = predictint(imvalue)
    print (predint[0]) #first value in list