from utils import *
import tensorflow as tf
from models import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('output_dim', 16, 'Output_dim of user final embedding.')
flags.DEFINE_integer('latent_dim', 32,'Latent_dim of user&item.')

# Load data
rating, concept, features, adjacency, negative = load_data()

# Some preprocessing
support = [adjacency]
num_support = len(support)

# Define placeholders
placeholders = {
    'rating': tf.placeholder(dtype=tf.float32,shape=rating.shape,name="rating"),
    'features': tf.placeholder(dtype=tf.float32,shape=features.shape,name='features'),
    'concept': tf.placeholder(dtype=tf.float32,shape=concept.shape,name="concept"),
    'support': [tf.placeholder(dtype=tf.float32,name='support'+str(_)) for _ in range(num_support)],
    'dropout': tf.placeholder_with_default(0.,shape=(),name='dropout'),
    'negative': tf.placeholder(dtype=tf.int32, shape=negative.shape, name='negative')
}

# Create model
model = GCN(placeholders=placeholders,input_dim=features.shape[1])

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

# Train model
for epoch in range(FLAGS.epochs):

    # Construct feed dictionary
    feed_dict = construct_feed_dict(placeholders,features,rating,concept,support, negative)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # if epoch%100 == 0:
    _, loss, hrat5 = sess.run([model.opt_op, model.loss, model.hrat5],feed_dict=feed_dict)
    if epoch%100 == 0:
        print('Train:'+str(epoch)+' Loss:'+str(loss)+' HR@5:'+str(hrat5))