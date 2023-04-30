import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import Task1A_E
import cv2
import os
from keras.callbacks import Callback

'''
Arguments - Image
Return - Image
Description - This function transforms the image to be fed into model and returns the preprocessed image
'''
def greek_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.array(img).astype("float32") / 255
    return img

'''
This derived class is used to terminate training when a parameter reacheres certain threshold
'''
class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='accuracy', baseline=1.0):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('\n \nEpoch %d: Reached accuracy of %d, terminating training' % (epoch,self.baseline))
                self.model.stop_training = True

'''
Arguments - None
Return - None
Description - This function starts execution and calls function to perform all sub-questions of Task 3
'''
def main():
    # Set seed and disbale CUDA
    np.random.seed(42)
    os.environ['TF_ENABLE_CUDNN'] = '0'
    
    # Preprocess and label training set
    folder = 'greek_training'
    class_names = ['alpha', 'beta', 'gamma'] # list of class names in the order you want them assigned
    train_images = []
    train_labels = []
    for filename in os.listdir(folder):
        train_labels.append(class_names.index(filename.split('_')[0]))
        img = cv2.imread(os.path.join(folder, filename))
        img = greek_transform(img)
        train_images.append(img)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Preprocess and label testing set
    test_images_actual = []
    folder = 'mygreek'
    class_names = ['alpha', 'beta', 'gamma'] # list of class names in the order you want them assigned
    test_images = []
    test_labels = []
    for filename in os.listdir(folder):
        test_labels.append(class_names.index(filename.split('_')[0]))
        img = cv2.imread(os.path.join(folder, filename))
        test_images_actual.append(img)
        img = greek_transform(img)
        test_images.append(img)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    
    train_labels = tf.keras.utils.to_categorical(train_labels, 3)
    test_labels = tf.keras.utils.to_categorical(test_labels, 3)
    
    # Create a base model and read weights of trained model into it
    base_model = Task1A_E.MyModel()
    base_model.load_weights('DNNweights').expect_partial()
    
    # Remove the last layer and its activation
    base_model.pop()
    base_model.pop()
    x = base_model.output
    
    # Add new last layer to new model
    preds = tf.keras.layers.Dense(3,activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=preds)
    model.summary()
    
    # Make layers of old model are untrainable
    for layer in model.layers:
        layer.trainable = False
    
    # Only last layer is trainable
    model.layers[-1].trainable = True
    
    # Build model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    es = [TerminateOnBaseline(monitor='accuracy', baseline=1.0)]
    history = model.fit(train_images, train_labels, epochs=10000, callbacks = es, batch_size = 64)
    
    # Make predictions on custom test set
    predictions = model.predict(test_images)
    
    # Plot predictions
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('New Input Predictions')
    for i in range(6):
        ax = axs[i//3, i%3]
        ax.imshow(test_images_actual[i])
        ax.set_title(f'Prediction:{class_names[tf.argmax(predictions[i])]}')
        ax.axis('off')
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.9,wspace=1,hspace=0.1)
    plt.show()
    
    # Plot loss curve for training
    plt.plot(history.history['loss'])
    plt.title('Loss Curve Task 3')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()
    
    return
    
if __name__ == "__main__":
   main()
