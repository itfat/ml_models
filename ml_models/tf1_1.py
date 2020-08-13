import tensorflow as tf
# Load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
# Convert the samples from integers to floating-point numbers
X_train,X_test=X_train / 255.0, X_test / 255.0
# Build the tf.keras.Sequential model by stacking layers.
# Choose an optimizer and loss function.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
predictions = model(X_train[:1]).numpy()
print(predictions)
# The tf.nn.softmax function converts these logits to "probabilities" for each class
print(tf.nn.softmax(predictions).numpy())
# The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example
loss_fn= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(Y_train[:1], predictions).numpy())
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
# The Model.fit method adjusts the model parameters to minimize the loss
model.fit(X_train,Y_train,epochs=5)
# The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set"
print(model.evaluate(X_test, Y_test, verbose=2))
# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it
probability_model= tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print(probability_model(X_test[:5]))