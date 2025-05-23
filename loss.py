import torch
from torch.autograd import Variable
import torch.nn.functional as F

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num,config, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.config=config
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1,1).to(self.config.device)  # 注意，这里的alpha是给定的一个list(tensor
        #),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1).to(self.config.device) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log().to(self.config.device)
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss