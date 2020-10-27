import torch
import numpy as np
from scipy.io.wavfile import write

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
tacotron2 = torch.hub.load(
    'nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()
