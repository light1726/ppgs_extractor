import tensorflow as tf
import numpy as np
import argparse
import time
import json
import os


DATA_DIR = '/data/data/vc_data/zhiling'
FID = '00000001'
SAVE_NAME = os.path.join('./test_results', FID + '.wav')
CKPT = './saved_models/wavenet.ckpt-73500'
PRIMER = False


def read_inputs(wav_path, receptive_fields):
    wav, _ = load_wav(wav_path, sr=None)

    return


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description="WaveNet training script")
    parser.add_argument('--fid', type=str, default=FID)
    parser.add_argument('--save_name', type=str, default=SAVE_NAME)
    parser.add_argument('--ckpt', type=str, default=CKPT)
    return parser.parse_args()


def main():
    args = get_arguments()
    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Set up network


    # set up a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    start_time = time.time()

    # load saved model
    saver = tf.train.Saver(tf.trainable_variables())
    # sess.run(tf.global_variables_initializer())
    print('Restoring model from {}'.format(args.ckpt))
    saver.restore(sess, args.ckpt)

    duration = time.time() - start_time
    print("Wav file generated in {:.3f} seconds".format(duration))
    sess.close()


if __name__ == '__main__':
    main()
