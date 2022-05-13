import time
import datetime
from train_pipeline3 import train_model
from configs3 import cfg
import torch
import random
import os
import numpy as np

def set_seed(seed=911):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


torch.autograd.set_detect_anomaly(True)
set_seed()

if __name__ == '__main__':
    start = time.time()
    train_model(cfg)
    end = time.time()
    t = str(datetime.timedelta(seconds=round(end-start)))
    print(f'Training done in {t} hours !!!')