import tensorflow as tf
# tf.logging.set_verboity(tf.logging.ERROR)
import numpy as np
celsius  = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenhiet = np.array([-40,14,32,46,59,72,100], dtype=float)
for i,c in enumerate(celsius):
    print("{} degree celsius ={} degree fahrenheit".format(c,fahrenhiet[i]))
#  First Way
# layers = tf.keras.layers.Dense(units=1, input_shape=[1])
# final_model = tf.keras.Sequential([layers])
# Another Way
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
histoy = model.fit(celsius, fahrenhiet, epochs=500, verbose=False)
print("Training is finished!!!")
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel('Loss Magnitude')
plt.plot(histoy.history['loss'])
plt.show()
print(model.predict([100]))
print('These are the layer variables: {}'.format(model.get_weights()))
