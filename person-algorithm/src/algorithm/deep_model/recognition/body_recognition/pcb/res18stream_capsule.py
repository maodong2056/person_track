from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num=2, activ='sigmoid', num_bottleneck=256):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Classifier(nn.Module):
    def __init__(self, num_feature=1024, dropout=0.25, num_classes=0):
        super(Classifier, self).__init__()
        self.dropout = dropout
        if dropout > 0:
            self.drop = nn.Dropout(dropout)
        if num_classes > 0:
            self.classifier = nn.Linear(num_feature, num_classes)
            init.normal_(self.classifier.weight, std=0.001)
            init.constant_(self.classifier.bias, 0)

    def forward(self, x, output_feature=None):

        if self.dropout > 0:
            x = self.drop(x)
        x = self.classifier(x)

        return x

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class DenseNet(nn.Module):
    def __init__(self, depth=121, num_feature=1024, num_classes=632, num_iteration = 3):

        super(DenseNet, self).__init__()
        self.depth = depth

        self.part = 4

        num_feature=512
        resnet= models.resnet18(pretrained=False)
        resnet.layer4[0].conv1.stride=(1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.maxpool, resnet.layer1,
                                  resnet.layer2, resnet.layer3, resnet.layer4)


        self.seblock = SELayer(channel=num_feature, reduction=16)
        # self.classifer1 = Classifier(num_feature=num_feature, dropout=0.25, num_classes=num_classes)
        # self.classifer2 = Classifier(num_feature=num_feature, dropout=0, num_classes=num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))



        self.gender_vagpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gender_classifer = ClassBlock(input_dim=num_feature, class_num=2, activ='none')




    def forward(self, x, output_feature=None):
        #x = self.base.features(x)#20.1024,7,7
        x = self.base(x)

        gender_y = F.avg_pool2d(x, x.size()[2:])
        gender_y = gender_y.view(gender_y.size(0), -1)

        gender_out_y = self.gender_classifer(gender_y)

        y1 = self.avgpool(x)

        if output_feature == 'pool5':
            y1 = y1.view(y1.size(0), y1.size(1), y1.size(2))

            ff = F.normalize(y1)
            # fnorm = torch.norm(ff,p=2,dim =1,keepdim=True)*np.sqrt(self.part)
            #
            # ff =ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0),-1)
            #y1 = F.normalize(y1)
            return gender_out_y,ff
