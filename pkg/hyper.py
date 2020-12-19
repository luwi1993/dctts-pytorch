import os


class Hyper:
    # audio
    audio_preemph = 0.97
    audio_nfft = 2048
    audio_samplerate = 22050
    audio_winlen = 0.05
    audio_winstep = 0.0125
    audio_melfilters = 80
    audio_refdB = 20
    audio_maxdB = 100
    audio_power = 1.5  # Exponent for amplifying the predicted magnitude
    audio_niter = 50  # Number of inversion iterations
    temporal_rate = 4

    # dir
    root_dir = "/Users/luwi/PycharmProjects/dctts-pytorch/"
    feat_dir = os.path.join(root_dir, "features")
    logdir = os.path.join(root_dir, "logdir")
    data_dir = "/Users/luwi/Documents/Datasets/Emotional_Speech/LJSpeech-1.1"
    
    # data
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: padding, E: end of string
    data_max_text_length = 200
    data_max_mel_length = 240

    # net
    dim_f = 80  # the dim of audio feature
    dim_e = 128
    dim_d = 256  # the hidden layer of Text2Mel
    dim_c = 512  # the hidder layer of SuperRes
    # dropout
    dropout = 0.05
    # train
    batch_size = 32
    num_batches = 1000000
    device_text2mel = "cpu:0"
    device_superres = "cpu:0"
    guide_g = 0.2  # bigger g, bigger guide area
    guide_weight = 100.0
    guide_decay = 0.99999
    guide_lowbound = 1

    # adam
    adam_alpha = 2e-4
    adam_betas = (0.5, 0.9)
    adam_eps = 1e-6
