#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import random
from keras.callbacks import TensorBoard


checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5' , verbose=1, save_best_only=True)

#training will stop if there is no change in contiune 10 epochs
callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),tf.keras.callbacks.TensorBoard(log_dir="logs")]
print(X_train.shape, Y_train.shape)

results = model.fit(X_train, Y_train, validation_split=0.05, batch_size=2, epochs=50, verbose=1,callbacks=callbacks)


preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.95):], verbose=1)
preds_test = model.predict(X_test, verbose=1)


preds_train_t = (preds_train > 0.5).astype(np.float32)
preds_val_t = (preds_val > 0.5).astype(np.float32)
preds_test_t = (preds_test > 0.5).astype(np.float32)


# In[ ]:


from matplotlib import pyplot

_, train_acc = model.evaluate(X_train, Y_train, verbose=1)
_, test_acc = model.evaluate(X_test, Y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#showing any rendom input image, ground truth, and output image
ix = 0
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

ix = 0
imshow(X_train[int(X_train.shape[0]*0.95):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.95):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()

ix = 0
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(Y_test[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()

pyplot.subplot(211)
pyplot.title('CloudFCN')
pyplot.plot(results.history['loss'], label='train')
pyplot.plot(results.history['val_loss'], label='test')
pyplot.ylabel("Loss")
pyplot.xlabel("Epochs")
pyplot.legend()

pyplot.subplot(212)
pyplot.plot(results.history['accuracy'], label='train')
pyplot.plot(results.history['val_accuracy'], label='test')
pyplot.ylabel("Accuracy")
pyplot.xlabel("Epochs")
pyplot.legend()
pyplot.show()

