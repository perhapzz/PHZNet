from torch import nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PHZNet(nn.Module):
    def __init__(self, in_features=16093, out_features=16093):
        super(PHZNet, self).__init__()
        # D: depth, L: segment length
        # x: input tensor of shape (batch_size, in_features)
        self.in_features = in_features
        self.out_features = out_features

        self.phznet = nn.Sequential(
            hybrid_zipped_network(in_features, 2),
            nn.Sigmoid()
        )


    def forward(self, input):
        # 将输入的numpy格式转换成tensor
        # out = torch.from_numpy(inputs).to(torch.float32)
        # print(inputs.shape)
        out = self.phznet(input)
        # print(out.shape)
        return out

# Hybrid Zippped Network
class hybrid_zipped_network(nn.Module):
    def __init__(self, in_features, out_features):
        super(hybrid_zipped_network, self).__init__()
        # D: depth, L: segment length
        # x: input tensor of shape ()
        self.in_features = in_features
        self.out_features = out_features
        self.seg_lengths = self.find_max_factors(in_features)

        self.hybrid_blocks = []
        for sl in self.seg_lengths:
            hb = hybrid_block(in_features, out_features, sl).to(device)
            self.hybrid_blocks.append(hb)

        self.norm_1 = nn.LayerNorm(self.in_features)
        self.linear_1 = nn.Linear(self.in_features, self.out_features)
        self.linear_2 = nn.Linear(self.in_features, self.out_features)

    def find_max_factors(self, n):
        factors = []
        for i in range(1, min(n+1, 21)):
            if n % i == 0:
                factors.append(i)
        factors.sort(reverse=True)
        return factors[:3] if len(factors) >= 3 else factors

    def forward(self, x):
        x = self.norm_1(x)
        output = self.linear_1(x)
        for hb in self.hybrid_blocks:
            output += hb(x)
        return output


# Hybrid Block
class hybrid_block(nn.Module):
    def __init__(self, in_features=16093, out_features=16093, seg_length=7):
        super(hybrid_block, self).__init__()
        # D: depth, L: segment length
        # x: input tensor of shape ()
        self.in_features = in_features
        self.out_features = out_features
        self.seg_length = seg_length

        self.proj_d = nn.Linear(self.seg_length, self.seg_length)
        self.proj_e = nn.Linear(self.seg_length, self.seg_length)

        self.linear = nn.Linear(self.in_features, self.out_features)

        # self.norm = nn.LayerNorm(self.in_features)
    
    def segmentation_linear_block(self, x):
        x_e = x.reshape(-1, self.in_features//self.seg_length, self.seg_length)
        x_e = self.proj_e(x_e)
        x_e = x_e.reshape(-1, self.in_features)
        return x_e

    def permute_segmentation_linear_block(self, x):
        x_d = x.reshape(-1, self.in_features//self.seg_length, self.seg_length)
        x_d = x_d.permute(0, 2, 1).reshape(-1, self.in_features//self.seg_length, self.seg_length)
        x_d = self.proj_d(x_d)
        x_d = x_d.reshape(-1, self.seg_length, self.in_features//self.seg_length).permute(0, 2, 1)
        x_d = x_d.reshape(-1, self.in_features)
        return x_d
    
    def forward(self, x):
        # output = self.linear(self.norm(x))
        # output = output + self.linear(self.segmentation_linear_block(self.norm(x)))
        # output = output + self.linear(self.permute_segmentation_linear_block(self.norm(x)))
        output = self.linear(x)
        output = output + self.linear(self.segmentation_linear_block(x))
        output = output + self.linear(self.permute_segmentation_linear_block(x))
        return output
