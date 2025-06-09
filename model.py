import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transformer import TransformerModel

class base_model(nn.Module):
    def __init__(
            self,
            model_name: str,
            pretrained_model_path: str,
            hidden_dim: int,
            dropout: float,
            class_n: int = 16,
            span_average: bool = False
    ):
        super().__init__()

        # Encoder

        self.transformer = TransformerModel(class_n, pretrained_model_path)
        self.dense = nn.Linear(self.transformer.model.config.hidden_size, hidden_dim)

        self.span_average = span_average
        self.classifier = nn.Linear(hidden_dim * 3, class_n)  # 修改输入维度
        self.layer_drop = nn.Dropout(dropout)

    def forward(self, inputs, weight=None):
        #############################################################################################
        # word representation，词输入信息
        input_token = inputs['bert_token']
        attention_mask = (input_token > 0).int()
        word_mapback = inputs['bert_word_mapback']
        token_length = inputs['token_length']
        length = inputs['bert_length']

        transformer_out = self.transformer(input_token, attention_mask=attention_mask)

        bert_seq_indi = sequence_mask(length).unsqueeze(dim=-1)
        transformer_out = transformer_out[:, 1:max(length) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)

        transformer_out = torch.bmm(word_mapback_one_hot.float(), self.dense(transformer_out))
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        transformer_out = transformer_out / wnt.unsqueeze(dim=-1)  # h_i

        #############################################################################################
        # span representation，span信息
        max_seq = transformer_out.shape[1]

        token_length_mask = sequence_mask(token_length)
        # 创建了一个上三角矩阵，用于表示候选标签之间的掩码
        candidate_tag_mask = torch.triu(torch.ones(max_seq, max_seq, dtype=torch.int64, device=transformer_out.device),
                                        diagonal=0).unsqueeze(dim=0) * (
                                         token_length_mask.unsqueeze(dim=1) * token_length_mask.unsqueeze(dim=-1))

        # 边界特征和跨度特征
        boundary_table_features = torch.cat(
            [transformer_out.unsqueeze(dim=2).repeat(1, 1, max_seq, 1), transformer_out.unsqueeze(dim=1)
            .repeat(1, max_seq, 1, 1)],
            dim=-1) * candidate_tag_mask.unsqueeze(dim=-1)  # h_i ; h_j
        span_table_features = form_raw_span_features(transformer_out, candidate_tag_mask,
                                                     is_average=self.span_average)  # sum(h_i,h_{i+1},...,h_{j})

        # h_i ; h_j ; sum(h_i,h_{i+1},...,h_{j})
        table_features = torch.cat([boundary_table_features, span_table_features], dim=-1)

        #############################################################################################
        # classifier
        logits = self.classifier(self.layer_drop(table_features)) * candidate_tag_mask.unsqueeze(dim=-1)

        # new
        outputs = {'logits': logits}

        if 'golden_label' in inputs:
            focal_loss = FocalLoss()(logits, inputs['golden_label'], weight, candidate_tag_mask)
            outputs['loss'] = focal_loss 
        #new

        return outputs


# 生成序列掩码，处理变长序列。返回值：一个布尔类型的二维张量，形状为[batch_size, max_len]，其中True表示序列的有效部分，False表示填充的部分。
def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) < (lengths.unsqueeze(1))


# 构建跨度特征，接受一个特征张量v，一个候选标签掩码candidate_tag_mask，以及一个布尔值is_average来决定是否对跨度特征进行平均。
def form_raw_span_features(v, candidate_tag_mask, is_average=True):
    new_v = v.unsqueeze(dim=1) * candidate_tag_mask.unsqueeze(dim=-1)
    span_features = torch.matmul(new_v.transpose(1, -1).transpose(2, -1),
                                 candidate_tag_mask.unsqueeze(dim=1).float()).transpose(2, 1).transpose(2, -1)

    if is_average:
        _, max_seq, _ = v.shape
        sub_v = (torch.tensor(range(1, max_seq + 1), device=v.device)
                 .unsqueeze(dim=-1) - torch.tensor(range(max_seq),device=v.device))
        sub_v = torch.where(sub_v > 0, sub_v, 1).T
        span_features = span_features / sub_v.unsqueeze(dim=0).unsqueeze(dim=-1)

    return span_features


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, weight=None, mask=None):
        # 计算交叉熵损失（带类别权重）
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none',
            weight=weight  # 注入类别权重
        )
        ce_loss = ce_loss.view(*targets.shape)  # 恢复为 (B, T)
        # 应用 Mask（仅计算有效位置的损失）
        if mask is not None:
            ce_loss = ce_loss * mask.float()

        # 计算 Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 仅对有效位置求平均
        if mask is not None:
            valid_count = mask.sum() + 1e-8
            return focal_loss.sum() / valid_count
        else:
            return focal_loss.mean()
        
