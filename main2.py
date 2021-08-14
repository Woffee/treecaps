"""

使用 big-vul 的数据集训练。
"""

from tensorflow import saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import os
import logging
import pickle
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from pathlib import Path
import numpy as np
import network as network
import sampling as sampling
import sys
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_loader import load_program_data
from data_loader import MonoLanguageProgramData
import argparse
import random
# import shutil
# import progressbar
from keras_radam.training import RAdamOptimizer
import time
import keras

# log file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = BASE_DIR + "/logs"
Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

now_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=LOG_PATH + '/' + now_time + '_prepare_data.log')
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', type=int, default=1, help='train batch size, always 1')
parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size, always 1')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--verbal', type=bool, default=True, help='print training info or not')
parser.add_argument('--n_classes', type=int, default=10, help='manual seed')

parser.add_argument('--train_directory', default="treecaps_data", help='train program data')
parser.add_argument('--model_path', default="model/batch_1", help='path to save the model')
parser.add_argument('--graphs_file', default="/data/function2vec4/graphs.pkl", help='')
parser.add_argument('--save_path', default="/data/function2vec4", help='')

parser.add_argument('--cache_path', default="cached", help='path to save the cache')
parser.add_argument('--test_directory', default="OJ_data/test", help='test program data')

parser.add_argument('--training', action="store_true", help='is training')
parser.add_argument('--testing', action="store_true",help='is testing')
parser.add_argument('--training_percentage', type=float, default=1.0 ,help='percentage of data use for training')
parser.add_argument('--log_path', default="" ,help='log path for tensorboard')
parser.add_argument('--epoch', type=int, default=0, help='epoch to test')

parser.add_argument('--cuda', default="0",type=str, help='enables cuda')

opt = parser.parse_args()
logger.info("treecaps parameters %s", opt)


os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

CASHED_PATH = opt.cache_path
Path(CASHED_PATH).mkdir(parents=True, exist_ok=True)

DATA_PATH = opt.train_directory
SAVE_PATH = opt.save_path

random.seed(9)


def train_model(train_trees, val_trees, labels, embedding_lookup, opt):
    max_acc = 0.0
    logdir = opt.model_path
    batch_size = opt.train_batch_size
    epochs = opt.niter
    
    # random.shuffle(train_trees)
    
    nodes_node, children_node, codecaps_node, codeCaps = network.init_net_treecaps(50, embedding_lookup, len(labels))

    codecaps_node = tf.identity(codecaps_node, name="codecaps_node")
    codeCaps = tf.identity(codeCaps, name="codecaps")

    out_node = network.out_layer(codecaps_node)
    labels_node, loss_node = network.loss_layer(codecaps_node, len(labels))

    optimizer = RAdamOptimizer(opt.lr)
    train_point = optimizer.minimize(loss_node)
    
     ### init the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))

    checkfile = os.path.join(logdir, 'tree_network.ckpt')

    print("Begin training..........")
    logger.info("Begin training..........")
    num_batches = len(train_trees) // batch_size + (1 if len(train_trees) % batch_size != 0 else 0)
    max_acc = 0.0
    for epoch in range(1, epochs+1):
       
        for train_step, train_batch in enumerate(sampling.batch_samples(
            sampling.gen_samples(train_trees, labels), batch_size
        )):
            nodes, children, batch_labels = train_batch
            # step = (epoch - 1) * num_batches + train_step * batch_size

            if not nodes:
                continue
            _, err, out, emb = sess.run(
                [train_point, loss_node, out_node, codeCaps],
                feed_dict={
                    nodes_node: nodes,
                    children_node: children,
                    labels_node: batch_labels
                }
            )
         
            print("Epoch : ", str(epoch), "Step : ", train_step, "Loss : ", err, "Max Acc: ",max_acc)
            logger.info("Epoch: {}, step: {}, loss: {}, max acc: {}".format(epoch, train_step, err, max_acc))


            if train_step % 1000 == 0 and train_step > 0:
                correct_labels = []
                predictions = []
                # logits = []
                for test_batch in sampling.batch_samples(
                    sampling.gen_samples(val_trees, labels), batch_size
                ):
                    print("---------------")
                    nodes, children, batch_labels = test_batch
                    print(batch_labels)
                    output = sess.run([out_node],
                        feed_dict={
                            nodes_node: nodes,
                            children_node: children
                        }
                    )

                    batch_correct_labels = np.argmax(batch_labels, axis=1)
                    batch_predictions = np.argmax(output[0], axis=1)
                    correct_labels.extend(batch_correct_labels)
                    predictions.extend(batch_predictions)
                    # logits.append(output)

                    print(batch_correct_labels)
                    print(batch_predictions)

                acc = accuracy_score(correct_labels, predictions)
                if (acc>max_acc):
                    max_acc = acc
                    saver.save(sess, checkfile)
                    print("Saved checkpoint....")

                print('Epoch',str(epoch),'Accuracy:', acc, 'Max Acc: ',max_acc)
                csv_log.write(str(epoch)+','+str(acc)+','+str(max_acc)+'\n')

    print("Finish all iters, storring the whole model..........")
    logger.info("Finish all iters, storring the whole model..........")


# def test_model(test_trees, labels, embeddings, embedding_lookup, opt):
    
    # logdir = opt.model_path
    # batch_size = opt.train_batch_size
    # epochs = opt.niter
    # num_feats = len(embeddings[0])

    # random.shuffle(test_trees)

    # # build the inputs and outputs of the network
    # nodes_node, children_node, codecaps_node = network.init_net_treecaps(num_feats,len(labels))
 
    # out_node = network.out_layer(codecaps_node)
    # labels_node, loss_node = network.loss_layer(codecaps_node, len(labels))

    # optimizer = RAdamOptimizer(opt.lr)
    # train_step = optimizer.minimize(loss_node)

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    # with tf.name_scope('saver'):
    #     saver = tf.train.Saver()
    #     ckpt = tf.train.get_checkpoint_state(logdir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         print("Continue training with old model")
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #         for i, var in enumerate(saver._var_list):
    #             print('Var {}: {}'.format(i, var))
                
    # checkfile = os.path.join(logdir, 'tree_network.ckpt')

    # correct_labels = []
    # predictions = []
    # print('Computing training accuracy...')
    # for batch in sampling.batch_samples(
    #     sampling.gen_samples(test_trees, labels, embeddings, embedding_lookup), 1
    # ):
    #     nodes, children, batch_labels = batch
    #     output = sess.run([out_node],
    #         feed_dict={
    #             nodes_node: nodes,
    #             children_node: children,
    #         }
    #     )
    #     correct_labels.append(np.argmax(batch_labels))
    #     predictions.append(np.argmax(output))

    # target_names = list(labels)
    # print(classification_report(correct_labels, predictions, target_names=target_names))
    # print(confusion_matrix(correct_labels, predictions))
    # print('*'*50)
    # print('Accuracy:', accuracy_score(correct_labels, predictions))
    # print('*'*50)


def predict(val_trees, labels, embedding_lookup, opt):
    logdir = opt.model_path
    batch_size = opt.train_batch_size
    epochs = opt.niter

    # random.shuffle(train_trees)

    nodes_node, children_node, codecaps_node, codeCaps = network.init_net_treecaps(50, embedding_lookup, len(labels))

    codecaps_node = tf.identity(codecaps_node, name="codecaps_node")
    codeCaps = tf.identity(codeCaps, name="codecaps")

    out_node = network.out_layer(codecaps_node)
    labels_node, loss_node = network.loss_layer(codecaps_node, len(labels))

    optimizer = RAdamOptimizer(opt.lr)
    train_point = optimizer.minimize(loss_node)

    ### init the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Loading model: {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

    correct_labels = []
    predictions = []
    # logits = []

    to_file = SAVE_PATH + "/all_func_embeddings_treecaps.pkl"
    embeddings = {}
    ii = 0
    for test_batch in sampling.batch_samples(
            sampling.gen_samples(val_trees, labels), batch_size
    ):
        # print("---------------")
        nodes, children, batch_labels, func_keys = test_batch

        if not nodes:
            continue

        # nodes = tf.convert_to_tensor(nodes, dtype=tf.float32)
        # print(batch_labels)
        # logger.info("out_node: {}, codeCaps: {}, nodes: {}, children: {}".format( type(out_node), type(codeCaps), type(nodes), type(children) ))

        logger.info("test_batch now: {}".format(ii))
        ii += 1

        # logger.info(" === nodes: {}".format(nodes))
        # logger.info(" === children: {}".format(children))

        # 注意：不要 把 codeEmb 命名为 codeCaps，否则会覆盖 codeCaps。 参见： https://stackoverflow.com/a/54855498
        output, codeEmb = sess.run([out_node, codeCaps],
                          feed_dict={
                              nodes_node: nodes,
                              children_node: children
                          }
                          )
        logger.info("codeEmb: {}".format(codeEmb.shape))
        # emb = keras.layers.flatten(codeCaps)
        emb = tf.reshape(codeEmb, shape=[1, 80])
        # logger.info("len(func_key): {}".format(len(func_key)))
        # logger.info("func_key: {}".format(func_key))
        embeddings[ func_keys[0] ] = emb
        # logger.info("emb: {}".format(emb.shape))
        # exit()
    with open(to_file, 'wb') as file_handler:
        pickle.dump(embeddings, file_handler, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("saved to: {}".format(to_file))

def main(opt):
    

    # with open(opt.node_type_lookup_path, 'rb') as fh:
    #     node_type_lookup = pickle.load(fh,encoding='latin1')
       
    labels = [str(i) for i in range(opt.n_classes)]

    if opt.training:
        print("Loading node type....")
        logger.info("Loading node type...")
        node_type_lookup_file = DATA_PATH + "/node_type_lookup.pkl"
        with open(node_type_lookup_file, 'rb') as fh:
            node_type_lookup = pickle.load(fh, encoding='latin1')

        logger.info("len of node_type_lookup: {}".format(len(node_type_lookup.keys())))


        print("Loading train trees...")
        logger.info("Loading train trees...")
        cached_path = opt.cache_path

        def get_nodes_num(node):
            res = 1
            for ch in node['children']:
                res += get_nodes_num(ch)
            return res


        all_trees = []
        for i in range(opt.n_classes):
            tree_file = DATA_PATH + "/training/treecaps_trees_{}.pkl".format(i)
            with open(tree_file, 'rb') as file_handler:
                trees = pickle.load(file_handler)
            for tree in trees:
                nn = get_nodes_num(tree['tree'])
                logger.info("== nn: {}".format(nn))
                if nn >= 25 and nn <= 500:
                    all_trees.append(tree)

        random.shuffle(all_trees)

        data_size = len(all_trees)
        logger.info("data size: {}".format(data_size))

        train_trees = all_trees[ : int(data_size * 0.8) ]
        val_trees = all_trees[ int(data_size * 0.8) : int(data_size * 0.9)]

        # train_data_loader = MonoLanguageProgramData(opt.train_directory, 0, opt.n_classes, cached_path)
        # train_trees, _ = train_data_loader.trees, train_data_loader.labels

        # val_data_loader = MonoLanguageProgramData(opt.test_directory, 2, opt.n_classes, cached_path)
        # val_trees, _ = val_data_loader.trees, val_data_loader.labels

        train_model(train_trees, val_trees, labels, node_type_lookup , opt) 

    if opt.testing:
        # /xye_data_nobackup/wenbo/dlvp/data/treecaps/data/all
        logger.info("start testing...")
        print("Loading node type....")
        logger.info("Loading node type...")
        node_type_lookup_file = DATA_PATH + "/node_type_lookup.pkl"
        with open(node_type_lookup_file, 'rb') as fh:
            node_type_lookup = pickle.load(fh, encoding='latin1')

        logger.info("len of node_type_lookup: {}".format(len(node_type_lookup.keys())))

        print("Loading train trees...")
        logger.info("Loading train trees...")
        cached_path = opt.cache_path

        def get_nodes_num(node):
            res = 1
            for ch in node['children']:
                res += get_nodes_num(ch)
            return res

        all_trees = []
        for i in range(308): # 共 308 个 project
            tree_file = DATA_PATH + "/all/treecaps_trees_{}.pkl".format(i)
            if not os.path.exists(tree_file):
                continue

            with open(tree_file, 'rb') as file_handler:
                trees = pickle.load(file_handler)
            for tree in trees:
                nn = get_nodes_num(tree['tree'])
                # logger.info("== nn: {}".format(nn))
                if nn >= 25 and nn <= 500:
                    all_trees.append(tree)
        logger.info("len(all_trees): {}".format(len(all_trees)))
        predict(all_trees, labels, node_type_lookup, opt)

if __name__ == "__main__":
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    csv_log = open(opt.model_path+'/log.csv', "w")
    csv_log.write('Epoch,Training Loss,Validation Accuracy\n')

    logger.info("opt: {}".format(opt))
    main(opt)
    csv_log.close()