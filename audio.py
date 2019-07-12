import librosa
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy import signal

hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 1024,
    'hop_length': 80,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  #
    'min_db': -80.0,  # restrict the dynamic range of log power
    'iterations': 100,  # griffin_lim #iterations
    'silence_db': -28.0,
}

_mel_basis = None


def load_wav(wav_f, sr=None):
    wav_arr, _ = librosa.load(wav_f, sr=sr)
    return wav_arr


def write_wav(write_path, wav_arr, sr):
    wav_arr *= 32767 / max(0.01, np.max(np.abs(wav_arr)))
    wavfile.write(write_path, sr, wav_arr.astype(np.int16))
    return


def preempahsis(wav_arr, pre_param=hparams['preemphasis']):
    return signal.lfilter([1, -pre_param], [1], wav_arr)


def deemphasis(wav_arr, pre_param=hparams['preemphasis']):
    return signal.lfilter([1], [1, -pre_param], wav_arr)


def split_wav(wav_arr, top_db=-hparams['silence_db']):
    intervals = librosa.effects.split(wav_arr, top_db=top_db)
    return intervals


def mulaw_encode(wav_arr, quantization_channels):
    mu = float(quantization_channels - 1)
    safe_wav_abs = np.minimum(np.abs(wav_arr), 1.0)
    encoded = np.sign(wav_arr) * np.log1p(mu * safe_wav_abs) / np.log1p(mu)
    return encoded


def mulaw_encode_quantize(wav_arr, quantization_channels):
    mu = float(quantization_channels - 1)
    safe_wav_abs = np.minimum(np.abs(wav_arr), 1.0)
    encoded = np.sign(wav_arr) * np.log1p(mu * safe_wav_abs) / np.log1p(mu)
    return ((encoded + 1.) / 2 * mu + 0.5).astype(np.int32)


def mulaw_decode(encoded, quantization_channels):
    mu = float(quantization_channels - 1)
    magnitude = (1 / mu) * ((1 + mu) ** abs(encoded) - 1.)
    return np.sign(encoded) * magnitude


def mulaw_decode_quantize(encoded, quantization_channels):
    mu = float(quantization_channels - 1)
    signal = 2 * (encoded.astype(np.float32) / mu) - 1.
    magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1.)
    return np.sign(signal) * magnitude


def mulaw_encode_quantize_tf(wav_batch, quantization_channels):
    with tf.variable_scope('mulaw_encode'):
        mu = tf.cast(quantization_channels - 1, tf.float32)
        safe_wav_abs = tf.minimum(tf.abs(wav_batch), 1.0)
        encoded = tf.sign(wav_batch) * tf.log1p(mu * safe_wav_abs) / tf.log1p(mu)
        return tf.cast((encoded + 1.) / 2 * mu + 0.5, tf.int32)


def mulaw_encode_tf(wav_batch, quantization_channels):
    with tf.variable_scope('mulaw_encode'):
        mu = tf.cast(quantization_channels - 1, tf.float32)
        safe_wav_abs = tf.minimum(tf.abs(wav_batch), 1.0)
        encoded = tf.sign(wav_batch) * tf.log1p(mu * safe_wav_abs) / tf.log1p(mu)
        return encoded


def mulaw_decode_quantize_tf(encoded, quantization_channels):
    with tf.variable_scope('mulaw_decode'):
        mu = tf.cast(quantization_channels - 1, tf.float32)
        signal = 2 * (tf.cast(encoded, tf.float32) / mu) - 1.
        magnitude = (1 / mu) * ((1 + mu) ** abs(signal) - 1.)
        return tf.sign(signal) * magnitude


def mulaw_decode_tf(encoded, quantization_channels):
    with tf.variable_scope('mulaw_decode'):
        mu = tf.cast(quantization_channels - 1, tf.float32)
        magnitude = (1 / mu) * ((1 + mu) ** abs(encoded) - 1.)
        return tf.sign(encoded) * magnitude


def stft(wav_arr, n_fft=hparams['n_fft'],
         hop_len=hparams['hop_length'],
         win_len=hparams['win_length'],
         window=hparams['window']):
    # return shape: [n_freqs, time]
    return librosa.core.stft(wav_arr, n_fft=n_fft, hop_length=hop_len,
                             win_length=win_len, window=window)


def stft_tf(wav_arr, n_fft=hparams['n_fft'],
            hop_len=hparams['hop_length'],
            win_len=hparams['win_length'],
            window=hparams['window']):
    window_f = {'hann': tf.contrib.signal.hann_window,
                'hamming': tf.contrib.signal.hamming_window}[window]
    # returned value is of shape [..., frames, fft_bins] and complex64 value
    return tf.contrib.signal.stft(signals=wav_arr, frame_length=win_len,
                                  frame_step=hop_len, fft_length=n_fft,
                                  window_fn=window_f)


def istft(stft_matrix, hop_len=hparams['hop_length'],
          win_len=hparams['win_length'], window=hparams['window']):
    # stft_matrix should be complex stft results instead of magnitude spectrogram
    # or power spectrogram, and of shape [n_freqs, time]
    return librosa.core.istft(stft_matrix, hop_length=hop_len,
                              win_length=win_len, window=window)


def istft_tf(stft_matrix, hop_len=hparams['hop_length'], n_fft=hparams['n_fft'],
             win_len=hparams['win_length'], window=hparams['window']):
    window_f = {'hann': tf.contrib.signal.hann_window,
                'hamming': tf.contrib.signal.hamming_window}[window]
    # stft_matrix should be of shape [..., frames, fft_bins]
    return tf.contrib.signal.inverse_stft(stft_matrix, frame_length=win_len,
                                          frame_step=hop_len, fft_length=n_fft,
                                          window_fn=window_f)


def spectrogram(wav_arr, n_fft=hparams['n_fft'],
                hop_len=hparams['hop_length'],
                win_len=hparams['win_length'],
                window=hparams['window']):
    # return shape: [time, n_freqs]
    s = stft(wav_arr, n_fft=n_fft, hop_len=hop_len,
             win_len=win_len, window=window).T
    magnitude = np.abs(s)
    power = magnitude ** 2
    return {'magnitude': magnitude,
            'power': power}


def power_spec2mel(power_spec, sr=hparams['sample_rate'], n_fft=hparams['n_fft'],
                   num_mels=hparams['num_mels'], fmin=hparams['fmin'], fmax=hparams['fmax']):
    # power_spec should be of shape [1+n_fft/2, time]
    global _mel_basis
    _mel_basis = (librosa.filters.mel(sr, n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
                  if _mel_basis is None else _mel_basis)  # [n_mels, 1+n_fft/2]
    mel_spec = np.dot(_mel_basis, power_spec)  # [n_mels, time]
    return mel_spec.T


def wav2melspec(wav_arr, sr=hparams['sample_rate'], n_fft=hparams['n_fft'],
                hop_len=hparams['hop_length'], win_len=hparams['win_length'],
                window=hparams['window'], num_mels=hparams['num_mels'],
                fmin=hparams['fmin'], fmax=hparams['fmax']):
    power_spec = spectrogram(wav_arr, n_fft, hop_len, win_len, window)['power']
    melspec = power_spec2mel(power_spec.T, sr, n_fft, num_mels, fmin, fmax)
    return melspec  # [time, num_mels]


def wav2mfcc(wav_arr, sr=hparams['sample_rate'], n_fft=hparams['n_fft'],
             hop_len=hparams['hop_length'], win_len=hparams['win_length'],
             window=hparams['window'], num_mels=hparams['num_mels'],
             fmin=hparams['fmin'], fmax=hparams['fmax']):


def mel2log_mel(mel_spec, ref_db=hparams['ref_db'], min_db=hparams['min_db']):
    log_mel = power2db(mel_spec, ref_db)
    normalized = log_power_normalize(log_mel, min_db)
    return normalized


def power2db(power_spec, ref_db=hparams['ref_db'], tol=1e-5):
    # power spectrogram is stft ** 2
    # returned value: (10. * log10(power_spec) - ref_db)
    return 10. * np.log10(power_spec + tol) - ref_db


def db2power(power_db, ref_db=hparams['ref_db']):
    return np.power(10.0, 0.1 * (power_db + ref_db))


def db2power_tf(power_db, ref_db=hparams['ref_db']):
    return tf.pow(10.0, 0.1 * (power_db + ref_db))


def log_power_normalize(log_power, min_db=hparams['min_db']):
    """
    :param log_power: in db, computed by power2db(spectrogram(wav_arr)['power'])
    :param min_db: minimum value of log_power in db
    :return: log_power normalized to [0., 1.]
    """
    assert min_db < 0. or "min_db should be a negative value like -80.0 or -100.0"
    return np.clip((log_power - min_db) / -min_db, 0., 1.)


def log_power_denormalize(normalized_logpower, min_db=hparams['min_db']):
    return np.clip(normalized_logpower, 0., 1.) * -min_db + min_db


def log_power_denormalize_tf(normalized_logpower, min_db=hparams['min_db']):
    return tf.clip_by_value(normalized_logpower, 0., 1.) * -min_db + min_db


def griffin_lim(magnitude_spec, iterations=hparams['iterations']):
    """
    :param magnitude_spec: magnitude spectrogram of shape [time, n_freqs]
                           obtained from spectrogram(wav_arr)['magnitude]
    :param iterations: number of iterations to estimate phase
    :return: waveform array
    """
    mag = magnitude_spec.T  # transpose to [n_freqs, time]
    angles = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    complex_mag = np.abs(mag).astype(np.complex)
    stft_0 = complex_mag * angles
    y = istft(stft_0)
    for i in range(iterations):
        angles = np.exp(1j * np.angle(stft(y)))
        y = istft(complex_mag * angles)
    return y


def grinffin_lim_tf(magnitude_spec, iterations=hparams['iterations']):
    # magnitude_spec: [frames, fft_bins], of type tf.float32
    angles = tf.cast(
        tf.exp(2j * np.pi * tf.cast(
            tf.random_uniform(
                tf.shape(magnitude_spec)),
            dtype=tf.complex64)),
        dtype=tf.complex64)
    complex_mag = tf.cast(tf.abs(magnitude_spec), tf.complex64)
    stft_0 = complex_mag * angles
    y = istft_tf(stft_0)
    for i in range(iterations):
        angles = tf.exp(1j * tf.cast(tf.angle(stft_tf(y)), tf.complex64))
        y = istft_tf(complex_mag * angles)
    return y


def griffin_lim_test(wav_f, n_fft=hparams['n_fft'],
                     hop_len=hparams['hop_length'],
                     win_len=hparams['win_length'],
                     window=hparams['window']):
    wav_arr = load_wav(wav_f)
    spec_dict = spectrogram(wav_arr, n_fft=n_fft, hop_len=hop_len,
                            win_len=win_len, window=window)
    mag_spec = spec_dict['magnitude']
    y = griffin_lim(mag_spec)
    write_wav('reconstructed1.wav', y, sr=16000)


def stft2wav_test(stft_f, mean_f, std_f):
    spec = np.load(stft_f)
    mean = np.load(mean_f)
    std = np.load(std_f)
    spec = spec * std + mean
    spec = log_power_denormalize(spec)
    power_spec = db2power(spec)
    mag_spec = power_spec ** 0.5
    y = griffin_lim(mag_spec)
    y = deemphasis(y)
    write_wav('reconstructed2.wav', y, sr=16000)
    return y


def stft2wav_tf_test(stft_f, mean_f, std_f):
    # get inputs
    spec = np.load(stft_f)
    mean = np.load(mean_f)
    std = np.load(std_f)
    spec = spec * std + mean
    # build graph
    spec_pl = tf.placeholder(tf.float32, [None, None, 513])
    denormalized = log_power_denormalize_tf(spec_pl)
    mag_spec = tf.pow(db2power_tf(denormalized), 0.5)
    wav = grinffin_lim_tf(mag_spec)
    # set session and run
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    wav_arr = sess.run(wav, feed_dict={spec_pl: np.expand_dims(spec, axis=0)})
    sess.close()
    y = deemphasis(np.squeeze(wav_arr))
    write_wav('reconstructed_tf.wav', y, sr=16000)
    return y
