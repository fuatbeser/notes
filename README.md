# notes
Useful techniques and methods that you can use while you are developing machine learning models.

# Stop training process if the trained model reaches the desired accuracy (or loss)
```python
DESIRED_ACCURACY = 0.98

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if(logs.get('acc') > DESIRED_ACCURACY):
      print("\nReached {}% accuracy so cancelling training!".format(DESIRED_ACCURACY*100)
      self.model.stop_training = True

callbacks = myCallback()
```

### How to use?

We can make use of [tf.keras.callbacks.Callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) to stop training if it reaches a desired accuracy (or loss). Simply create a `myCallback` class and call it.  After that, don't forget to add `[callbacks]`.

```python
model.fit(training_data, training_labels, epochs=100, callbacks=[callbacks])
```

If you would like to use `loss` instead of `accuracy`, use `logs.get('loss'`.
