import os
import math
import numpy as np
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from models.efficientnet import EfficientNet


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    data_root = "/home/liuhuijun/Datasets/ImageNet"
    val_dir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    arch = "b7"
    img_preparam = {"b0": (224, 0.875),
                    "b1": (240, 0.882),
                    "b2": (260, 0.890),
                    "b3": (300, 0.904),
                    "b4": (380, 0.922),
                    "b5": (456, 0.934),
                    "b6": (528, 0.942),
                    "b7": (600, 0.949)}
    valid_dataset = datasets.ImageFolder(val_dir, transforms.Compose([transforms.Resize(int(img_preparam[arch][0] / img_preparam[arch][1]), Image.BICUBIC),
                                                                      transforms.CenterCrop(img_preparam[arch][0]),
                                                                      transforms.ToTensor(),
                                                                      normalize]))

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False,
                                               num_workers=16, pin_memory=False)
    num_batches = int(math.ceil(len(valid_loader.dataset) / float(valid_loader.batch_size)))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model = EfficientNet(arch=arch, num_classes=1000).cuda()
    used_gpus = [idx for idx in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model, device_ids=used_gpus).cuda()

    checkpoint = torch.load("/home/liuhuijun/TrainLog/release/imagenet/efficientnet_{}_top1v_86.7.pkl".format(arch))
    pre_weight = checkpoint['model_state']
    model_dict = model.state_dict()
    pretrained_dict = {"module." + k: v for k, v in pre_weight.items() if "module." + k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        pbar = tqdm(np.arange(num_batches))
        for i_val, (images, labels) in enumerate(valid_loader):

            images = images.cuda()
            labels = torch.squeeze(labels.cuda())

            net_out = model(images)

            prec1, prec5 = accuracy(net_out, labels, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            pbar.update(1)
            pbar.set_description("> Eval")
            pbar.set_postfix(Top1=top1.avg, Top5=top5.avg)
        pbar.set_postfix(Top1=top1.avg, Top5=top5.avg)
        pbar.update(1)
        pbar.close()
