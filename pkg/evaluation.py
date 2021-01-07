from pkg.synthesis import synthesis as synthesize
from pkg.hyper import Hyper
import numpy as np
from pkg.data import load_vocab
import codecs
import os
from pkg.utils import text_normalize
import pandas as pd


class Evaluator:
    def __init__(self, keys=["index", "init_time", "Latency_beginning", "Latency_synthesis", "duration_mels",
                             "duration_mags", "duration_total", "file_name", "n_samples", "duration",
                             "roboticness", "gpu_util", "max_memory_required", "repetitions", "skipping",
                             "time_measurements", "relative_synthesis_time"]):
        self.keys = keys
        self.log = {key: [] for key in self.keys}

    # hardware Measures
    def gpu_util(self):
        return 0

    def max_memory_required(self):
        return 0

    # Phenomenon Detection
    def pausenlänge(self):
        pass

    def roboticness(self):
        return 0

    def repetitions(self):
        return 0

    def skipping(self):
        return 0

    # Loss Measures
    def autokorrelation(self, pred, target):
        pass

    def mel_cepstral_distortion(self, pred, target):
        K = 10 / np.log(10) * np.sqrt(2)
        return K * np.mean(np.sqrt(np.sum((pred - target) ** 2, axis=1)))

    def calculate_f0(self, x):
        pass

    def rmse_for_f0(self, pred, target):
        pass

    # Evaluatio
    def evaluate_time_measurements(self, time_measurements):
        for key in time_measurements.keys():
            self.log[key].append(time_measurements[key])

    def load_test_set(self, load_n):
        char2idx, _ = load_vocab()

        csv = os.path.join(Hyper.test_data_dir, "metadata.csv")
        names, lengths, texts = [], [], []
        with codecs.open(csv, 'r', "utf-8") as f:
            lines = f.readlines()
            for line in lines[:load_n]:
                line = line.strip()
                fname, _, text = line.split('|')
                text = text_normalize(text) + 'E'  # append the end of string mark
                for char in text:
                    if char <= '9' and char >= '0':
                        raise ValueError("[data]: after text normalize, there should be no digits.")
                # text = [char2idx[char] for char in text]

                names.append(fname)
                lengths.append(len(text))
                texts.append(text)
                if len(text) > Hyper.data_max_text_length:
                    raise Exception("[load data]: length of text is out of range")

        return names, lengths, texts

    def load_outside_domain_texts(self):
        pass

    def evaluate(self, epoch):
        names, lengths, texts = self.load_test_set(5)
        info = synthesize(texts)
        for file_name in info["samples"].keys():
            self.log["epoch"].append(epoch)
            utterance = info["samples"][file_name]
            self.log["file_name"].append(file_name)
            self.log["n_samples"].append(len(utterance))
            self.log["duration"].append(self.log["n_samples"][-1] / Hyper.audio_samplerate)

            time_measurements = info["time_measurements"]
            self.evaluate_time_measurements(time_measurements)
            self.log["relative_synthesis_time"].append(self.log["duration"][-1] / self.log["duration_total"][-1])

    def export(self):
        max_len = [len(self.log[key]) for key in self.log.keys()]
        valid_keys = [key for key in self.log.keys() if len(self.log[key]) == max_len]
        pd.DataFrame(self.log[valid_keys]).to_csv("/Users/luwi/PycharmProjects/dctts-pytorch/log.csv", sep="\t")
