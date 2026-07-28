import torch
import torch.nn.functional as F
import torch
import torch.nn as nn

class classify_using_cosine_similarity(nn.Module):
    def __init__(self,):
        super(classify_using_cosine_similarity, self).__init__()


        # 由于视觉特征和文本特征维度不同(1536 vs 768)，需要映射到相同维度
        # 创建一个线性层进行维度转换
        self.projection = torch.nn.Linear(1536, 768, bias=False)

    def forward(self, visual_features, text_features):
        """
        根据余弦相似性对视觉特征和文本特征进行分类

        参数:
            visual_features: 视觉特征张量，形状为 (B, 1536, 12, 12)
                             B为批次大小，1536为特征通道数，12x12为空间维度
            text_features: 文本特征张量，形状为 (Nc, 768)
                           Nc为类别数量，768为文本特征维度

        返回:
            predictions: 预测的类别索引，形状为 (B,)
        """
        # 获取批次大小和类别数量
        # B = visual_features.shape[0]
        # Nc = text_features.shape[0]

        # 对视觉特征进行空间平均池化，得到 (B, 1536)
        # 这一步将12x12的空间维度压缩为1，保留通道维度
        visual_embeddings = torch.mean(visual_features, dim=[2, 3])  # 结果形状: (B, 1536)

        # 将视觉特征投影到与文本特征相同的维度空间
        visual_projected = self.projection(visual_embeddings)  # 结果形状: (B, 768)

        # 对特征进行归一化，为计算余弦相似度做准备
        visual_norm = F.normalize(visual_projected, p=2, dim=1)  # 形状: (B, 768)
        text_norm = F.normalize(text_features, p=2, dim=1)  # 形状: (Nc, 768)

        # 计算余弦相似度: (B, 768) 与 (Nc, 768) 计算得到 (B, Nc)
        # 等价于矩阵乘法(因为已经归一化)
        similarity_matrix = torch.matmul(visual_norm, text_norm.t())  # 形状: (B, Nc)

        # 找到每个样本相似度最高的类别索引
        predictions = torch.argmax(similarity_matrix, dim=1)  # 形状: (B,)

        return predictions


# 示例用法
if __name__ == "__main__":
    # 创建随机的视觉特征和文本特征作为示例
    batch_size = 8
    num_classes = 10

    # 生成随机视觉特征 (B, 1536, 12, 12)
    visual_features = torch.randn(batch_size, 1536, 12, 12)
    # 生成随机文本特征 (Nc, 768)
    text_features = torch.randn(num_classes, 768)

    method = classify_using_cosine_similarity()
    # 进行分类预测
    predictions = method(visual_features, text_features)

    print(f"预测类别: {predictions}")
    print(f"预测类别形状: {predictions.shape}")
