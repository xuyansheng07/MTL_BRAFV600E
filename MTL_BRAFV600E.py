import torch
import torch.nn as nn
import numpy as np

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        out = avg_out + max_out
        return torch.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)
        return self.sigmoid(attention) * x


def Conv1(in_planes, places, stride=2):

    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class localFeature(nn.Module):
    def __init__(self,blocks=[3, 4, 6, 3], expansion = 4):
        super().__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.att1 = ChannelAttention(in_channels=256)

        self.layer2 = self.make_layer(in_places=256 ,places=128, block=blocks[1], stride=2)
        self.att2 = ChannelAttention(in_channels=512)

        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.att3 = ChannelAttention(in_channels=1024)

        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
        self.att4 = ChannelAttention(in_channels=2048)

        self.conv1x1_1 = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(2048, 2048, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x1 = self.layer1(x)
        x1 = self.att1(x1)

        x2 = self.layer2(x1)
        x2 = self.att2(x2)

        x3 = self.layer3(x2)
        x3 = self.att3(x3)

        x4 = self.layer4(x3)
        x4 = self.att4(x4)

        out1 = self.conv1x1_1(x2)
        out2 = self.conv1x1_2(x3)
        out3 = self.conv1x1_3(x4)

        return [out1,out2,out3]

class globalFeature(nn.Module):
    def __init__(self,blocks=[3, 4, 6, 3], expansion = 4):
        super().__init__()
        self.expansion = expansion

        self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.att1 = SpatialAttention()

        self.layer2 = self.make_layer(in_places=256 ,places=128, block=blocks[1], stride=2)
        self.att2 = SpatialAttention()

        self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
        self.att3 = SpatialAttention()

        self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)
        self.att4 = SpatialAttention()

        self.conv1x1_1 = nn.Conv2d(512, 2048, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(2048, 2048, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)

        x1 = self.layer1(x)
        x1 = self.att1(x1)

        x2 = self.layer2(x1)
        x2 = self.att2(x2)

        x3 = self.layer3(x2)
        x3 = self.att3(x3)

        x4 = self.layer4(x3)
        x4 = self.att4(x4)  #bc,2048,7,7

        out1 = self.conv1x1_1(x2)
        out2 = self.conv1x1_2(x3)
        out3 = self.conv1x1_3(x4)

        return [out1,out2,out3]


class LocalFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.localfeature = localFeature()

    def forward(self, x):

        x = self.localfeature(x)

        return x

class GlobalFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.glocalfeature = globalFeature()

    def forward(self, x):
        x = self.glocalfeature(x)

        return x

class Gate(nn.Module):
    def __init__(self, n_expert=3):
        super(Gate, self).__init__()

        # Convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, n_expert, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_expert),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)

        return x

class FeatureSelection(nn.Module):
    def __init__(self, local_channels,global_channels,r=4):
        super(FeatureSelection, self).__init__()
        self.r = r

        self.conv1 = nn.Sequential(
                                nn.Conv2d(local_channels, (local_channels+global_channels) // 2, kernel_size=1,stride=1),
                                nn.BatchNorm2d((local_channels+global_channels) // 2),
                                nn.ReLU(inplace=True)
                                   )

        self.conv2 = nn.Sequential(
                                nn.Conv2d(global_channels, (local_channels+global_channels) // 2, kernel_size=1, stride=1),
                                nn.BatchNorm2d((local_channels+global_channels) // 2),
                                nn.ReLU(inplace=True)
                                   )
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv3 = nn.Sequential(
                                nn.Conv2d((local_channels+global_channels), (local_channels+global_channels) // self.r, 1, 1, bias=False),
                                nn.BatchNorm2d((local_channels+global_channels) // self.r),
                                nn.ReLU(inplace=True),
                                nn.Conv2d((local_channels + global_channels)//self.r,(local_channels + global_channels) // 2, 1, 1),
                                nn.Softmax(dim=1)
                                    )

    def forward(self,l,g):
        att = self.conv3(self.gap(torch.cat((self.conv1(l),self.conv2(g)),dim=1)))
        output = l * att + g * (1 - att)
        return output


class Expert_Gate(nn.Module):

    def __init__(self,n_expert,output_channels):
        super(Expert_Gate, self).__init__()
        self.n_task = 5
        self.n_expert = n_expert
        self.output_channels = output_channels

        '''local feature network'''
        self.local_feature = LocalFeature()

        '''global feature network'''
        self.global_feature = GlobalFeature()

        '''feature selection'''
        for i in range(self.n_expert):
            setattr(self, "feature_selection_layer"+str(i + 1), FeatureSelection(self.output_channels,self.output_channels))
        self.feature_selection_layer = [getattr(self, "feature_selection_layer"+str(i + 1)) for i in range(self.n_expert)]

        self.avgpool2 = nn.AdaptiveAvgPool2d(1)

        for i in range(self.n_task):
            setattr(self, "gate_layer" + str(i + 1), Gate(3))

        self.gate_layers = [getattr(self, "gate_layer" + str(i + 1)) for i in range(self.n_task)]

    def forward(self, x):

        local_E_net = self.local_feature(x)

        global_E_net = self.global_feature(x)

        feature_net = [f for f in self.feature_selection_layer]

        gate_net = [gate(x) for gate in self.gate_layers]

        towers = []

        features = []
        for j in range(self.n_expert):

            local_feature = local_E_net[j]

            global_feature = global_E_net[j]

            f = feature_net[j](local_feature, global_feature)

            f1 = self.avgpool2(f)
            f2 = f1.view(f1.size(0), -1)

            features.append(f2)

        feature_E_net = torch.cat(([e[:, np.newaxis, :] for e in features]), dim=1)

        for i in range(self.n_task):
            g = gate_net[i].unsqueeze(2)
            tower = torch.matmul(feature_E_net.transpose(1, 2), g)
            towers.append(tower.transpose(1, 2).squeeze(1))


        return towers

class FusionGate(nn.Module):

    def __init__(self,n_feature):

        super(FusionGate, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, n_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_feature),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.softmax(x)

        return x


class MTL_BRAFV600E(nn.Module):

    def __init__(self, output_dim=2048, n_expert=3):

        '''

        :param expert_dim:
        :param n_expert:

        '''

        super(MTL_BRAFV600E, self).__init__()

        self.Expert_Gate = Expert_Gate(n_expert,output_dim)

        '''composition'''
        self.tower1 = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )
        '''echo'''
        self.tower2 = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 3),
        )
        '''margin'''
        self.tower3 = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )
        '''foci'''
        self.tower4 = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 5),
        )
        '''shape'''
        self.tower5 = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )
        '''brafv600e'''
        self.tower6 = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )

        self.gate = FusionGate(n_feature=5)

    def forward(self, x):

        towers = self.Expert_Gate(x)

        g = self.gate(x).unsqueeze(2)
        f = [towers[0], towers[1], towers[2], towers[3], towers[4]]
        feature = torch.cat(([e[:, np.newaxis, :] for e in f]), dim=1)
        feature_main = torch.matmul(feature.transpose(1, 2), g).transpose(1, 2).squeeze(1)
        out1 = self.tower1(towers[0])
        out2 = self.tower2(towers[1])
        out3 = self.tower3(towers[2])
        out4 = self.tower4(towers[3])
        out5 = self.tower5(towers[4])
        out = self.tower6(feature_main)


        return [out1, out2, out3, out4, out5, out]

    def named_weight_parameters(self):
        return filter(lambda x: not x[0].endswith('bias'),
                      self.named_parameters())

    def weight_parameters(self):
        for name, param in self.named_weight_parameters():
            yield param

    def named_bias_parameters(self):
        return filter(lambda x: x[0].endswith('bias'),
                      self.named_parameters())

    def weight_parameters(self):
        for name, param in self.named_bias_parameters():
            yield param


