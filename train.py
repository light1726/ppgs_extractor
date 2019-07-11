import tensorflow as tf
import argparse
from datetime import datetime
import time
import os
import sys
import numpy as np

from models import DNNClassifier

# some super parameters
BATCH_SIZE = 128
STEPS = int(1e6)
LEARNING_RATE = 4e-4
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MAX_TO_SAVE = 10
CKPT_EVERY = 500
MFCC_DIM = 39
PPG_DIM = 131
DATA_DIR = ''


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description="WaveNet training script")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--restore_from', type=str, default=None)
    parser.add_argument('--overwrite', type=_str_to_bool, default=True,
                        help='Whether to overwrite the old model ckpt,'
                             'valid when restore_from is not None')
    parser.add_argument('--max_ckpts', type=int, default=MAX_TO_SAVE)
    parser.add_argument('--ckpt_every', type=int, default=CKPT_EVERY)
    return parser.parse_args()


def save_model(saver, sess, logdir, step):
    model_name = 'vqvae.ckpt'
    ckpt_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, ckpt_path, global_step=step)
    print('Done')


def load_model(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root, mode='train'):
    logdir = os.path.join(logdir_root, mode, STARTED_DATESTRING)
    return logdir


def validate_directories(restore_dir, overwrite):
    if restore_dir is None:
        logdir = get_default_logdir('./logdir')
        dev_dir = get_default_logdir('./logdir', 'dev')
        restore_dir = logdir
        os.makedirs(logdir)
    elif overwrite:
        logdir = restore_dir
        dev_dir = restore_dir.replace('train', 'dev')
        if not os.path.isdir(logdir):
            raise ValueError('No such directory: {}'.format(restore_dir))
    else:
        logdir = get_default_logdir('./logdir')
        dev_dir = get_default_logdir('./logdir', 'dev')
        os.makedirs(logdir)
    return {'logdir': logdir, 'restore_from': restore_dir, 'dev_dir': dev_dir}


def main():
    args = get_arguments()

    directories = validate_directories(args.restore_from, args.overwrite)
    restore_dir = directories['restore_from']
    logdir = directories['logdir']
    dev_dir = directories['dev_dir']

    # load data
    train_mfccs = np.load(os.path.join(DATA_DIR, 'train_data_noisy.npy'))
    train_phonemes = np.load(os.path.join(DATA_DIR, 'train_label.npy'))
    train_len = train_mfccs.shape[0]
    with np.load(os.path.join(DATA_DIR, 'test_data_noisy.npy')) as data:
        time_len = data.shape[0]
        dev_mfccs = data[:time_len // 2, :]
        test_mfccs = data[time_len // 2:, :]
    with np.load(os.path.join(DATA_DIR, 'test_label.npy')) as data:
        time_len = data.shape[0]
        dev_phonemes = data[:time_len // 2, :]
        test_phonemes = data[time_len // 2:, :]
    dev_len = dev_mfccs.shape[0]
    test_len = test_mfccs.shape[0]

    mfcc_pl = tf.placeholder(dtype=tf.float32, shape=[None, MFCC_DIM], name='mfcc_pl')
    phonemes_pl = tf.placeholder(dtype=tf.int32, shape=[None, PPG_DIM], name='phoneme_pl')

    # create network and optimization operation
    classifier = DNNClassifier(out_dims=PPG_DIM, hiddens=[256, 256, 256, 256], name='dnn_classifier')

    results_dict = classifier(mfcc_pl, phonemes_pl)
    predicted = tf.nn.softmax(results_dict['logits'])
    tf.summary.image('predicted', predicted)
    tf.summary.image('groundtruth', phonemes_pl)
    loss = results_dict['cross_entropy']
    learning_rate_pl = tf.placeholder(tf.float32, None, 'learning_rate')
    tf.summary.scalar('cross_entropy', loss)
    tf.summary.scalar('learning_rate', learning_rate_pl)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_pl, epsilon=1e-4)
    optim = optimizer.minimize(loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optim = tf.group([optim, update_ops])

    # Set up logging for TensorBoard.
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(tf.get_default_graph())
    dev_writer = tf.summary.FileWriter(dev_dir)
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=args.max_ckpts)

    # set up session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    # try to load saved model
    try:
        saved_global_step = load_model(saver, sess, restore_dir)
        if saved_global_step is None:
            saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    last_saved_step = saved_global_step
    step = None
    try:
        for step in range(saved_global_step + 1, args.steps):
            train_inputs = train_mfccs[step % train_len: step % train_len + args.batch_size, :]
            train_labels = train_phonemes[step % train_len: step % train_len + args.batch_size, :]
            dev_inputs = dev_mfccs[step % dev_len: step % train_len + args.batch_size, :]
            dev_labels = dev_phonemes[step % dev_len: step % train_len + args.batch_size, :]
            if step <= int(4e5):
                lr = args.lr
            elif step <= int(6e5):
                lr = 0.5 * args.lr
            elif step <= int(8e5):
                lr = 0.25 * args.lr
            else:
                lr = 0.125 * args.lr
            start_time = time.time()
            if step % args.ckpt_every == 0:
                summary, loss_value = sess.run([summaries, loss],
                                               feed_dict={mfcc_pl: dev_inputs,
                                                          phonemes_pl: dev_labels,
                                                          learning_rate_pl: lr})
                dev_writer.add_summary(summary, step)
                duration = time.time() - start_time
                print('step {:d} - eval loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))
                save_model(saver, sess, logdir, step)
                last_saved_step = step
            else:
                summary, loss_value, _ = sess.run([summaries, loss, optim],
                                                  feed_dict={mfcc_pl: train_inputs,
                                                             phonemes_pl: train_labels,
                                                             learning_rate_pl: lr})
                train_writer.add_summary(summary, step)
                duration = time.time() - start_time
                print('step {:d} - training loss = {:.3f}, ({:.3f} sec/step)'
                      .format(step, loss_value, duration))
    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save_model(saver, sess, logdir, step)
    sess.close()


if __name__ == '__main__':
    main()
