import tensorflow as tf
import tensornets as nets

inputs=tf.placeholder(tf.float32, [None, 224, 224, 3])

outputs=tf.placeholder(tf.float32, [None, 7]) # nr of classes

is_train=tf.placeholder_with_default(False, shape=(), name="is_train") # placeholder for is_training
 
model=nets.VGG16(inputs, is_training=is_train, classes=7)
train_list=model.get_weights() # get list of weights
loss=tf.losses.softmax_cross_entropy(outputs, model)
  
accuracy, accuracy_op=tf.metrics.accuracy(tf.argmax(outputs, 1),tf.argmax(inputs,1)) # local vars
 
update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS) # update batch stats during training
with tf.control_dependencies(update_ops):               # only train last "block"
    train=tf.train.AdamOptimizer(1e-5).minimize(loss, train_list[520:]) 
  
init_op=tf.global_variables_initializer()
local_init_op=tf.local_variables_initializer()
 
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    sess.run(model.pretrained())
     
    for epoch in range(10):
       for (x, y) in train_data:  
           # run training
           sess.run(train, feed_dict={inputs: x, outputs: y, is_train: True})
       for (x, y) in test_data:
           # run testing 
           sess.run(accuracy_op, feed_dict={inputs: x, outputs: y, is_train: False}) # use global stats for batch norm
