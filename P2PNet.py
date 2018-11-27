import os
import sys
import collections
# add pointnet++ dir path
BASE_PATH = os.path.dirname(__file__)
sys.path.append(BASE_PATH)
sys.path.append(BASE_PATH + "/pointnet++/tf_ops")
sys.path.append(BASE_PATH + "/pointnet++/tf_ops/3d_interpolation")
sys.path.append(BASE_PATH + "/pointnet++/tf_ops/grouping")
sys.path.append(BASE_PATH + "/pointnet++/tf_ops/sampling")
sys.path.append(BASE_PATH + "/pointnet++/utils")

import numpy as np
import tensorflow as tf

# import pointnet++ set abstraction and feature propagation modules
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module


#  our network model
Model = collections.namedtuple("Model", \
                               "pointSetX_ph,  pointSetY_ph, \
                               istraining_ph,\
                               PredictedX, PredictedY, \
                               dataLossX, shapeLossX, densityLossX, \
                               dataLossY, shapeLossY, densityLossY, \
                               regul_loss, \
                               data_train, total_train, \
                               learning_rate,  global_step,  bn_decay, \
                               training_sum_ops, testing_sum_ops,\
                               train_dataloss_A_ph,  train_dataloss_B_ph, train_regul_ph, \
                               test_dataloss_A_ph,   test_dataloss_B_ph,  test_regul_ph"     )



def create_model(FLAGS):
  # Initialize model training hyperparameters
  # -----------------------------------------

  global_step = tf.train.get_or_create_global_step()
  
  learning_rate = tf.train.exponential_decay(
    0.001, # base learning rate
    global_step * FLAGS.batch_size, # number of steps
    FLAGS.example_count * FLAGS.decayEpoch, # step size, decayEpoch: steps (how many epochs) for decaying learning rate
    0.5, # decay rate
    staircase = True
  )
  learning_rate = tf.maximum(learning_rate, 1e-4)

  bn_momentum = tf.train.exponential_decay(
    0.5, # base momentum parameter
    global_step * FLAGS.batch_size, # number of steps
    FLAGS.example_count * FLAGS.decayEpoch * 2, # step size
    0.5,
    staircase = True
  )
  bn_decay = tf.minimum(0.99, 1 - bn_momentum)


  # Create the network
  # -----------------------------------------
  
  # placeholders
  pointsetX_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
  pointsetY_ph = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.point_num, 3))
  istraining_ph = tf.placeholder(tf.bool, shape=())
  
  # noise for augmentation
  noiseX = None
  noiseY = None
  if FLAGS.noiseLength > 0:
    noiseX = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1.0, dtype=tf.float32)
    noiseY = tf.random_normal(shape=[FLAGS.batch_size, FLAGS.point_num, FLAGS.noiseLength], mean=0.0, stddev=1.0, dtype=tf.float32)
  
  # initialize two net(X->Y and Y->X)
  with tf.variable_scope("P2PNet_X2Y") as scope:
    displaceX2Y = get_displacements(pointsetX_ph, istraining_ph, noiseX, FLAGS, bn_decay)
  
  with tf.variable_scope("P2PNet_Y2X") as scope:
    displaceY2X = get_displacements(pointsetY_ph, istraining_ph, noiseY, FLAGS, bn_decay)

  # predicted results
  predictedSetX = pointsetX_ph + displaceX2Y
  predictedSetY = pointsetY_ph + displaceY2X

  # get geometric loss (shape loss, density loss)
  dataLossX, shapeLossX, densityLossX = get_geometric_loss(predictedSetX, pointsetX_ph, FLAGS)
  dataLossY, shapeLossY, densityLossY = get_geometric_loss(predictedSetY, pointsetY_ph, FLAGS)

  # Regularizatiom
  if FLAGS.regular_weight > 0.0:
    regularLoss = get_regularizing_loss(pointsetX_ph, pointsetY_ph, predictedSetX, predictedSetY)
  else:
    regularLoss = tf.constant(0.0, dtype=tf.float32)

  # total loss
  dataLoss = dataLossX + dataLossY
  totalLoss = dataLoss + regularLoss * FLAGS.regular_weight

  # get the train result by minimizing the loss
  trainVariables = tf.trainable_variables()
  trainer = tf.train.AdamOptimizer(learning_rate)

  dataTrainOp = trainer.minimize(dataLoss, var_list=trainVariables, global_step=global_step)
  totalTrainOp = trainer.minimize(totalLoss, var_list=trainVariables, global_step=global_step)

  dataTrain  = dataTrainOp
  totalTrain = totalTrainOp

  
  # Create the summarizers
  # -----------------------------------------

  # declare placeholders
  train_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
  train_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
  train_regular_ph = tf.placeholder(tf.float32, shape=())

  test_dataloss_A_ph = tf.placeholder(tf.float32, shape=())
  test_dataloss_B_ph = tf.placeholder(tf.float32, shape=())
  test_regular_ph = tf.placeholder(tf.float32, shape=())

  # using summary to record the scalars
  lr_sum_op = tf.summary.scalar('learning rate', learning_rate)
  global_step_sum_op = tf.summary.scalar('batch_number', global_step)

  train_dataloss_A_sum_op = tf.summary.scalar('train_dataloss_A', train_dataloss_A_ph)
  train_dataloss_B_sum_op = tf.summary.scalar('train_dataloss_B', train_dataloss_B_ph)
  train_regul_sum_op = tf.summary.scalar('train_regul', train_regular_ph)

  test_dataloss_A_sum_op = tf.summary.scalar('test_dataloss_A', test_dataloss_A_ph)
  test_dataloss_B_sum_op = tf.summary.scalar('test_dataloss_B', test_dataloss_B_ph)
  test_regul_sum_op = tf.summary.scalar('test_regul', test_regular_ph)

  # merge the scalar summary op to train summary and test summary
  training_sum_ops = tf.summary.merge( \
        [lr_sum_op, train_dataloss_A_sum_op, train_dataloss_B_sum_op, train_regul_sum_op])

  testing_sum_ops = tf.summary.merge( \
        [test_dataloss_A_sum_op, test_dataloss_B_sum_op, test_regul_sum_op ])


  # Finally, we return the model we get
  # -----------------------------------------
  return Model(
        pointsetX_ph=pointsetX_ph,  pointsetY_ph=pointsetY_ph,
        istraining_ph=istraining_ph,
        PredictedX=predictedSetX,   PredictedY=predictedSetY,
        dataLossX=dataLossX,   shapeLossX=shapeLossX,     densityLossX=densityLossX,
        dataLossY=dataLossY,   shapeLossY=shapeLossY,     densityLoss_B=densityLossY,
        regul_loss=regularLoss,
        data_train=dataTrain,     total_train=totalTrain,
        learning_rate=learning_rate, global_step=global_step, bn_decay=bn_decay,
        training_sum_ops=training_sum_ops, testing_sum_ops=testing_sum_ops,
        train_dataloss_A_ph=train_dataloss_A_ph, train_dataloss_B_ph=train_dataloss_B_ph, train_regul_ph=train_regular_ph, \
        test_dataloss_A_ph=test_dataloss_A_ph, test_dataloss_B_ph=test_dataloss_B_ph, test_regul_ph=test_regular_ph
    )


# get the displacements preoduced by pointnet++ abstraction set layer,
# feature propagation layer and out fully-connected layer
def get_displacements(pointset, istraining, noise, FLAGS, bn_decay=None):
  batch_size = FLAGS.batch_size
  num_points = FLAGS.point_num

  point_cloud = pointset

  # initial states of pointnet++
  l0_xyz = point_cloud
  l0_points = None

  # pointnet++ set abstraction layer
  l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1 * FLAGS.radiusScal, nsample=64,
                                                       mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                       is_training=istraining, bn_decay=bn_decay, scope='layer1')
  l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=384, radius=0.2 * FLAGS.radiusScal, nsample=64,
                                                      mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                      is_training=istraining, bn_decay=bn_decay, scope='layer2')
  l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.4 * FLAGS.radiusScal, nsample=64,
                                                      mlp=[256, 256, 512], mlp2=None, group_all=False,
                                                      is_training=istraining, bn_decay=bn_decay, scope='layer3')
  # output the pointnet by set abstraction layer
  l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
                                                       mlp=[512, 512, 1024], mlp2=None, group_all=True,
                                                       is_training=istraining, bn_decay=bn_decay, scope='layer4')

  # Feature Propagation layers
  l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [512, 512], istraining, bn_decay, scope='fa_layer1')
  l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [512, 256], istraining, bn_decay, scope='fa_layer2')
  l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], istraining, bn_decay, scope='fa_layer3')
  l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], istraining, bn_decay, scope='fa_layer4')

  # noise augmentation
  if noise is not None:
        l0_points = tf.concat(axis=2, values=[l0_points, noise])

  # fully connected layer
  net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=istraining, scope='fc1', bn_decay=bn_decay )
  net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=True, is_training=istraining, scope='fc2', bn_decay=bn_decay)
  net = tf_util.conv1d(net, 3, 1, padding='VALID', activation_fn=None, scope='fc3')

  # displacements
  displacements = tf.sigmoid(net) * FLAGS.range_max * 2 - FLAGS.range_max
  return displacements

def get_geometric_loss(predictedSet, sourceSet, FLAGS):

    # calculate shape loss
    square_dist = pairwise_l2_norm2_batch(sourceSet, predictedSet)
    dist = tf.sqrt( square_dist )
    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    shapeLoss = tf.reduce_mean(minRow) + tf.reduce_mean(minCol)
	
    # calculate density loss
    square_dist2 = pairwise_l2_norm2_batch(sourceSet, sourceSet)
    dist2 = tf.sqrt(square_dist2)
    knndis = tf.nn.top_k(tf.negative(dist), k=FLAGS.nnk)
    knndis2 = tf.nn.top_k(tf.negative(dist2), k=FLAGS.nnk)
    densityLoss = tf.reduce_mean(tf.abs(knndis.values - knndis2.values))
  
    # total data loss
    data_loss = shapeLoss + densityLoss * FLAGS.densityWeight
    return data_loss, shapeLoss, densityLoss


def get_regularizing_loss(pointSetX, pointSetY, predictedSetX, predictedSetY):

    displacementsX = tf.concat(axis=2, values=[pointSetX, predictedSetY])
    displacementsY = tf.concat(axis=2, values=[predictedSetX,   pointSetY])

    square_dist = pairwise_l2_norm2_batch( displacementsX,   displacementsY )
    dist = tf.sqrt(square_dist)

    minRow = tf.reduce_min(dist, axis=2)
    minCol = tf.reduce_min(dist, axis=1)
    RegularLoss = (tf.reduce_mean(minRow) + tf.reduce_mean(minCol))/2

    return RegularLoss

def pairwise_l2_norm2_batch(x, y, scope=None):
    with tf.op_scope([x, y], scope, 'pairwise_l2_norm2_batch'):
        nump_x = tf.shape(x)[1]
        nump_y = tf.shape(y)[1]

        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, nump_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, nump_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 2)

        return square_dist