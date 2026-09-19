
import torch
import torch.nn as nn

from einops import rearrange, repeat

class ESP_Fi_MLP(nn.Module):
    def __init__(self, num_classes):
        super(ESP_Fi_MLP, self).__init__()
        input_size = 1 * 950 * 52  
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),

        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x



class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  
        )
        

        self.feature_size = 128 * 118 * 6  
        

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)        
        x = x.view(x.size(0), -1)   
        x = self.classifier(x)      
        return x
    


# ------------------------------
# BasicBlock
# ------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=(1,1), i_downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.i_downsample is not None:
            identity = self.i_downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ------------------------------
# Bottleneck
# ------------------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=(1,1), i_downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.i_downsample is not None:
            identity = self.i_downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ------------------------------
# ESP-Fi ResNet
# ------------------------------
class ESP_Fi_ResNet(nn.Module):
    def __init__(self, ResBlock, layers, num_classes):
        super(ESP_Fi_ResNet, self).__init__()
        self.in_channels = 64


        self.preprocess = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=1, bias=False),  # 保持尺寸
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # 200x52 -> 100x26
        )

        self.layer1 = self._make_layer(ResBlock, 64, layers[0], stride=(1,1))
        self.layer2 = self._make_layer(ResBlock, 128, layers[1], stride=(2,1))  
        self.layer3 = self._make_layer(ResBlock, 256, layers[2], stride=(2,1))
        self.layer4 = self._make_layer(ResBlock, 512, layers[3], stride=(2,1))

        # 全局池化 + 全连接
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=(1,1)):
        i_downsample = None
        layers = []

        if stride != (1,1) or self.in_channels != out_channels * block.expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers.append(block(self.in_channels, out_channels, stride=stride, i_downsample=i_downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

def ESP_Fi_ResNet18(num_classes=7):
    return ESP_Fi_ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)


class ESP_Fi_ResNet18_MultiLink(nn.Module):
    """
    Multi-link extension of ESP_Fi_ResNet18 for `num_links` independent CSI
    streams (e.g. multiple STA-AP links/antennas), each shaped (1, 950, 52) --
    the same shape the single-link ESP_Fi_ResNet18 was pretrained on.

    The backbone (conv1 through the pooled 512-d feature) is weight-tied
    across all links and can be initialized directly from a single-link
    ESP_Fi_ResNet18 checkpoint with no modification -- see
    `load_pretrained_backbone`. Only the fusion layer + classification head
    are new and randomly initialized.

    Input: (B, num_links, 1, 950, 52) or (B, num_links, 950, 52).

    For a warm-start schedule (train the new fusion head with the pretrained
    backbone frozen, then unfreeze for end-to-end fine-tuning), use
    `freeze_backbone()` / `unfreeze_backbone()`. These also pin the backbone's
    BatchNorm layers in eval mode while frozen, so their running mean/var
    statistics aren't disturbed by the new, differently-distributed multi-link
    data during the frozen phase -- `requires_grad=False` alone stops weight
    updates but does NOT stop BatchNorm from updating its running stats.
    """

    def __init__(self, num_classes, num_links=3, freeze_backbone=False):
        super().__init__()
        self.num_links = num_links
        self._backbone_frozen = False

        self.backbone = ESP_Fi_ResNet18(num_classes=num_classes)
        feat_dim = self.backbone.fc.in_features  # 512 for BasicBlock (expansion=1)
        self.backbone.fc = nn.Identity()  # discard single-link classifier, keep pooled features

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * num_links, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Stop the backbone from training: disables gradients on its params
        and pins it in eval mode (so BatchNorm running stats stay fixed too).
        Call before building the optimizer so frozen params are excluded, or
        filter with `trainable_parameters()`.
        """
        self._backbone_frozen = True
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        return self

    def unfreeze_backbone(self):
        """Resume training the backbone end-to-end. Rebuild the optimizer
        afterwards so the newly-trainable params are included.
        """
        self._backbone_frozen = False
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train()
        return self

    def trainable_parameters(self):
        """Params with requires_grad=True -- pass this to the optimizer
        instead of self.parameters() so a frozen backbone isn't given
        zero-but-tracked gradients (wasted memory/compute) or, worse,
        nudged by weight decay alone.
        """
        return (p for p in self.parameters() if p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()  # keep BatchNorm stats frozen regardless of outer train()/eval()
        return self

    def load_pretrained_backbone(self, checkpoint_path, map_location="cpu"):
        """Load a single-link ESP_Fi_ResNet18 checkpoint (as saved by run.py's
        `torch.save(model.state_dict(), best_model_path)`) into the shared
        backbone. The checkpoint's `fc.*` weights are discarded (the fusion
        head replaces them); every other key must match exactly.
        """
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/backbone mismatch -- missing: {missing}, unexpected: {unexpected}"
            )
        return self

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, L, 950, 52) -> (B, L, 1, 950, 52)
        assert x.size(1) == self.num_links, f"expected {self.num_links} links, got {x.size(1)}"

        feats = [self.backbone(x[:, i]) for i in range(self.num_links)]  # weight-tied backbone reused per link
        fused = torch.cat(feats, dim=1)  # (B, feat_dim * num_links)
        return self.classifier(fused)


class ESP_Fi_GRU(nn.Module):
    def __init__(self, num_classes):
        super(ESP_Fi_GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=52,      
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, 1, 950, 52]
        x = x.squeeze(1)        # [B, 950, 52]

        output, ht = self.gru(x)

        feat = ht[-1]           # [B, 128]
        outputs = self.fc(feat)

        return outputs

    
    
    
class ESP_Fi_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=52,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.squeeze(1)          
        x = x[:, ::4, :]          

        output, _ = self.lstm(x)  
        feat = output.mean(dim=1)
        return self.fc(feat)





class TimePatchEmbedding(nn.Module):

    def __init__(self, in_channels=1, patch_size_t=50, emb_size=64):
        super().__init__()
        self.patch_size_t = patch_size_t
        self.emb_size = emb_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=emb_size,
            kernel_size=(1, patch_size_t),
            stride=(1, patch_size_t)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = None  

    def forward(self, x):
        B, C, H, T = x.shape
        device = x.device

        num_patches = T // self.patch_size_t

        x = self.proj(x)                          # [B, emb, H, num_patches]
        x = rearrange(x, 'b c h p -> b p h c')    # [B, p, h, c]
        x = rearrange(x, 'b p h c -> b (p h) c')  # [B, p*h, emb]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B).to(device)
        x = torch.cat([cls_tokens, x], dim=1)

        seq_len = x.shape[1]
        if self.pos_embed is None or self.pos_embed.shape[1] != seq_len:
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.emb_size)).to(device)

        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_size=64, num_heads=8, ff_mult=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_size)

        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_mult * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):


        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out

        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size=64, num_classes=7, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        cls = x[:, 0]           
        cls = self.norm(cls)
        cls = self.dropout(cls)
        return self.fc(cls)


class ESP_Fi_Transformer(nn.Module):

    def __init__(self,
                 num_classes=7,
                 patch_size_t=50,    
                 emb_size=64,
                 depth=4,               
                 num_heads=8,
                 ff_mult=4,
                 dropout=0.2):
        super().__init__()

        self.patch_embed = TimePatchEmbedding(
            in_channels=1,
            patch_size_t=patch_size_t,
            emb_size=emb_size
        )

        self.encoder = nn.ModuleList([
            TransformerBlock(
                emb_size=emb_size,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout
            ) for _ in range(depth)
        ])

        self.head = ClassificationHead(
            emb_size=emb_size,
            num_classes=num_classes,
            dropout=dropout + 0.1  # 分类头 dropout 稍大
        )

    def forward(self, x):
        x = self.patch_embed(x)

        for block in self.encoder:
            x = block(x)

        logits = self.head(x)
        return logits


    




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp_size, se=True, nl='HS'):
        super(MobileNetV3Block, self).__init__()
        self.stride = stride
        self.se = se

        self.conv1 = nn.Conv2d(in_channels, exp_size, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.nl1 = h_swish() if nl == 'HS' else nn.ReLU(inplace=True)

        if isinstance(kernel_size, tuple):
            padding = tuple((k - 1) // 2 for k in kernel_size)
        else:
            padding = (kernel_size - 1) // 2

        self.conv2 = nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, 
                               padding=padding, groups=exp_size, bias=False)
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.nl2 = h_swish() if nl == 'HS' else nn.ReLU(inplace=True)

        if se:
            self.se_module = SEModule(exp_size)

        self.conv3 = nn.Conv2d(exp_size, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x

        out = self.nl1(self.bn1(self.conv1(x)))

        out = self.nl2(self.bn2(self.conv2(out)))
        

        if self.se:
            out = self.se_module(out)
 
        out = self.bn3(self.conv3(out))
        
        if self.shortcut is not None:
            out += residual
        
        return out

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16),
            h_swish()
        )
        
        self.blocks = nn.Sequential(
            MobileNetV3Block(16, 16, kernel_size=(3, 3), stride=(1, 1), exp_size=16, se=False, nl='RE'),
            MobileNetV3Block(16, 24, kernel_size=(3, 3), stride=(2, 2), exp_size=72, se=False, nl='RE'),
            MobileNetV3Block(24, 24, kernel_size=(3, 3), stride=(1, 1), exp_size=88, se=False, nl='RE'),
            MobileNetV3Block(24, 40, kernel_size=(5, 5), stride=(2, 2), exp_size=96, se=True, nl='HS'),
            MobileNetV3Block(40, 40, kernel_size=(5, 5), stride=(1, 1), exp_size=240, se=True, nl='HS'),
            MobileNetV3Block(40, 40, kernel_size=(5, 5), stride=(1, 1), exp_size=240, se=True, nl='HS'),
            MobileNetV3Block(40, 48, kernel_size=(5, 5), stride=(1, 1), exp_size=120, se=True, nl='HS'),
            MobileNetV3Block(48, 48, kernel_size=(5, 5), stride=(1, 1), exp_size=144, se=True, nl='HS'),
            MobileNetV3Block(48, 96, kernel_size=(5, 5), stride=(2, 2), exp_size=288, se=True, nl='HS'),
            MobileNetV3Block(96, 96, kernel_size=(5, 5), stride=(1, 1), exp_size=576, se=True, nl='HS'),
            MobileNetV3Block(96, 96, kernel_size=(5, 5), stride=(1, 1), exp_size=576, se=True, nl='HS'),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(576),
            h_swish(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(576, 1280),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)  
        x = self.blocks(x)  
        x = self.head(x)  
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)  
        return x






class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            SiLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se=True):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.se = se
        
        expand_channels = in_channels * expand_ratio
        self.conv1 = nn.Conv2d(in_channels, expand_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_channels)
        self.act1 = SiLU()
        

        if isinstance(kernel_size, tuple):
            padding = tuple((k - 1) // 2 for k in kernel_size)
        else:
            padding = (kernel_size - 1) // 2

        self.conv2 = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, groups=expand_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_channels)
        self.act2 = SiLU()
        

        if se:
            self.se_module = SEModule(expand_channels)
        self.conv3 = nn.Conv2d(expand_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        
 
        out = self.act1(self.bn1(self.conv1(x)))
        
  
        out = self.act2(self.bn2(self.conv2(out)))
        

        if self.se:
            out = self.se_module(out)
        
  
        out = self.bn3(self.conv3(out))
        
 
        if self.shortcut is not None:
            out += residual
        
        return out


class EfficientNetLite(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super(EfficientNetLite, self).__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            SiLU()
        )
        self.blocks = nn.Sequential(

            InvertedResidualBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1, se=False),
            InvertedResidualBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6, se=False),
            InvertedResidualBlock(24, 24, kernel_size=3, stride=1, expand_ratio=6, se=False),
            

            InvertedResidualBlock(24, 40, kernel_size=5, stride=2, expand_ratio=6, se=True),
            InvertedResidualBlock(40, 40, kernel_size=5, stride=1, expand_ratio=6, se=True),
            

            InvertedResidualBlock(40, 80, kernel_size=3, stride=2, expand_ratio=6, se=True),
            InvertedResidualBlock(80, 80, kernel_size=3, stride=1, expand_ratio=6, se=True),
            InvertedResidualBlock(80, 112, kernel_size=5, stride=1, expand_ratio=6, se=True),
            InvertedResidualBlock(112, 112, kernel_size=5, stride=1, expand_ratio=6, se=True),
            InvertedResidualBlock(112, 192, kernel_size=5, stride=2, expand_ratio=6, se=True),
            InvertedResidualBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6, se=True),
            InvertedResidualBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6, se=True),
            InvertedResidualBlock(192, 320, kernel_size=3, stride=1, expand_ratio=6, se=True)
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            SiLU(),
            nn.AdaptiveAvgPool2d(1)  
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.stem(x)  
        
        x = self.blocks(x)  
        
        x = self.head(x)  
        
        x = x.view(x.size(0), -1)  
        
        x = self.classifier(x) 
        
        return x
