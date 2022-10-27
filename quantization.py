import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchvision
from torchvision import models
from torchvision import transforms
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import get_default_qconfig
from torch.utils.data import DataLoader
from modules.model import GeneratorFullModel
from modules.keypoint_detector import KPDetector
import yaml
from frames_dataset import FramesDataset
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from frames_dataset import DatasetRepeater


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image in data_loader:
            model(image)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    checkpoint = torch.load('./vox-cpk.pth.tar')
    # download the vox-cpk.pth.tar at https://drive.google.com/drive/folders/1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH
    with open('config/vox-256.yaml') as f:
        config = yaml.safe_load(f)
    dataset = FramesDataset(is_train=False, **config['dataset_params'])
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if torch.cuda.is_available():
        kp_detector.to(cuda_device)
    kp_detector.load_state_dict(checkpoint['kp_detector'])        
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if torch.cuda.is_available():
        generator.to(cuda_device)
    generator.load_state_dict(checkpoint['generator'])

    train_params = config['train_params'] 
    
    model_fp32 = GeneratorFullModel(kp_detector, generator, None, train_params)
    
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    
    model_fp32.eval()
    model_to_quantize = copy.deepcopy(model_fp32)
    model_to_quantize.eval()

    qconfig = get_default_qconfig("fbgemm")

    qconfig_dict = {"": qconfig}

    prepared_model = prepare_fx(model_to_quantize, qconfig_dict)、
    print(prepared_model.graph)

    # calibrate(prepared_model, data_loader)
    
    quantized_model = convert_fx(prepared_model)

    # torch.jit.save(torch.jit.script(quant_model), 'outQuant.pth') 



