import os
import mmcv

import torch


from common import N_CLASSES
from utils.metrics import runningScore


class Eval(object):

    def __init__(self, logdir):

        cfg_path = os.path.join(logdir, 'cfg.py')
        model_path = os.path.join(logdir, 'model.pth')

        self.device = torch.device('cuda')
        self.cfg = mmcv.Config.fromfile(cfg_path)

        ####################
        # 定义模型
        ####################
        from models.cccmodel import MobileDual
        self.model = MobileDual(n_classes=N_CLASSES)
        self.model.to(self.device)
        # 加载预训练模型
        self.model.load_state_dict(torch.load(model_path))

        ####################
        # 导入训练集、测试集
        ####################
        from torch.utils.data import DataLoader
        from data.irseg import IRSeg
        test_dataset = IRSeg(self.cfg, mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=0,
                                 pin_memory=True, drop_last=False)

        self.running_metrics_test = runningScore(n_classes=N_CLASSES)


    def eval(self):
        with torch.no_grad():
            self.model.eval()
            self.running_metrics_test.reset()
            for i, sample in enumerate(self.test_loader):
                image = sample['img'].to(self.device)
                depth = sample['dep'].to(self.device)
                label = sample['label_cls'].to(self.device)
                predict = self.model(image, depth)

                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                self.running_metrics_test.update(label, predict)

        meanRst, ius, accs = self.running_metrics_test.get_scores()
        print(meanRst)
        print(ius)
        print(accs)

    def show(self):
        pass




if __name__ == '__main__':
    logdir = r'E:\work\Master_Seg\Space_seg\run\Seg_model_baseline_v1_2022-06-13-19-23'

    valer = Eval(logdir)

    valer.eval()
    # valer.show()
