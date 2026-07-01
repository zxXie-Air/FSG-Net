import torch
import torch.nn as nn

class CrossAttentionWithTemporalEmbedding(nn.Module):
    """
    Synergistic Temporal-Spatial Attention Module (STSAM) - Branch 1: Cross-Attention

    Implements the cross-attention mechanism with temporal embeddings as described in the text.
    - Adds learnable temporal embeddings to input features.
    - Uses true cross-attention: Query from one temporal instance attends to Key/Value of the other.
    - `CrossAtt_1 = Softmax(Q_2 @ K_1^T) @ V_1`
    - `CrossAtt_2 = Softmax(Q_1 @ K_2^T) @ V_2`
    - Includes a residual connection.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # As per the text, Q and K channels are compressed.
        # To ensure matrix multiplication compatibility for Q_i @ K_j^T,
        # their channel dimensions must be equal. We'll use C/8 for both.
        self.q_channels = in_channels // 8
        self.k_channels = in_channels // 8
        self.v_channels = in_channels

        # Learnable temporal embeddings
        self.t_embedding1 = nn.Parameter(torch.randn(1, self.in_channels, 1, 1), requires_grad=True)
        self.t_embedding2 = nn.Parameter(torch.randn(1, self.in_channels, 1, 1), requires_grad=True)

        # Shared convolution layers for Q, K, V generation
        self.query_conv = nn.Conv2d(self.in_channels, self.q_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(self.in_channels, self.k_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(self.in_channels, self.v_channels, kernel_size=1)

        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)

        # Learnable scaling parameter for the residual connection
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        """
        f1: Feature map from time 1 (e.g., F_1)
        f2: Feature map from time 2 (e.g., F_2)
        """
        b, c, h, w = f1.shape

        # 1. Add temporal embeddings
        # The embeddings are broadcasted and added to the features.
        f1_emb = f1 + self.t_embedding1
        f2_emb = f2 + self.t_embedding2

        # 2. Generate Q, K, V for both temporal inputs using shared layers
        # For f1
        q1 = self.query_conv(f1_emb).view(b, -1, h * w).permute(0, 2, 1)  # (B, N, C_q)
        k1 = self.key_conv(f1_emb).view(b, -1, h * w)  # (B, C_k, N)
        v1 = self.value_conv(f1_emb).view(b, -1, h * w)  # (B, C_v, N)

        # For f2
        q2 = self.query_conv(f2_emb).view(b, -1, h * w).permute(0, 2, 1)  # (B, N, C_q)
        k2 = self.key_conv(f2_emb).view(b, -1, h * w)  # (B, C_k, N)
        v2 = self.value_conv(f2_emb).view(b, -1, h * w)  # (B, C_v, N)

        # 3. Perform Cross-Attention

        # 3.1. Enhance f1 using information from f2 (Q_2 queries K_1, V_1)
        # energy_21: (B, N, C_q) @ (B, C_k, N).transpose -> (B, N, C_q) @ (B, N, C_k) -> (B, N, N)
        energy_21 = torch.bmm(q2, k1)
        attn_21 = self.softmax(energy_21)
        # out1: (B, C_v, N) @ (B, N, N).transpose -> (B, C_v, N) @ (B, N, N) -> (B, C_v, N)
        out1 = torch.bmm(v1, attn_21.transpose(-2, -1))
        out1 = out1.view(b, c, h, w)

        # 3.2. Enhance f2 using information from f1 (Q_1 queries K_2, V_2)
        energy_12 = torch.bmm(q1, k2)
        attn_12 = self.softmax(energy_12)
        out2 = torch.bmm(v2, attn_12.transpose(-2, -1))
        out2 = out2.view(b, c, h, w)

        # 4. Apply residual connection to the original features
        cross_att1 = self.gamma * out1 + f1
        cross_att2 = self.gamma * out2 + f2

        return cross_att1, cross_att2


class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super(CoordAtt, self).__init__()

        # Adaptive average pooling layers for encoding spatial information
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pool along the width axis
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # Pool along the height axis

        # Calculate the intermediate channel dimension for the bottleneck
        mip = max(8, channels // reduction)

        # The 1x1 convolution block for fusing information and reducing channels
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        # Using ReLU as the non-linear activation function
        self.act = nn.ReLU(inplace=True)

        # Separate 1x1 convolutions to generate attention weights for each spatial direction
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 1. Coordinate Encoding
        # Pool input feature x along width and height directions respectively.
        x_h = self.pool_h(x)  # Shape: [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # Shape after permute: [n, c, w, 1]

        # 2. Attention Generation
        # Concatenate along the spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # Shape: [n, c, h+w, 1]

        # Fuse information using the shared convolution block
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split the fused feature back into horizontal and vertical components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # Restore shape for width component: [n, mip, 1, w]

        # Generate attention weights using separate convolutions and a sigmoid function
        a_h = self.conv_h(x_h).sigmoid()  # Shape: [n, c, h, 1]
        a_w = self.conv_w(x_w).sigmoid()  # Shape: [n, c, 1, w]

        # 3. Apply Attention
        # The attention weights are broadcasted and multiplied with the input feature map
        out = identity * a_w * a_h

        return out

class STSAM(nn.Module):

    def __init__(self, channels, reduction=32):
        super(STSAM, self).__init__()
        self.channels = channels

        # Instantiate Branch 1: Cross-Attention
        self.branch1 = CrossAttentionWithTemporalEmbedding(in_channels=channels)

        # Instantiate Branch 2: Coordinate Attention
        self.branch2 = CoordAtt(channels=channels, reduction=reduction)

        # Final fusion layer as described in the text
        # Input channels are 2*C because we concatenate the outputs of two branches
        # Output channels are C to restore the original feature dimension
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        """
        f1: Feature map from time 1, shape [B, C, H, W]
        f2: Feature map from time 2, shape [B, C, H, W]
        """
        # --- Parallel Processing ---

        # 1. Process through Branch 1 (Cross-Attention)
        # This branch handles the interaction between f1 and f2.
        f1_from_b1, f2_from_b1 = self.branch1(f1, f2)

        # 2. Process through Branch 2 (Coordinate Attention)
        # This branch processes f1 and f2 independently to enhance spatial features.
        f1_from_b2 = self.branch2(f1)
        f2_from_b2 = self.branch2(f2)

        # --- Fusion Stage ---

        # 3. Fuse the outputs for the first temporal feature (f1)
        f1_cat = torch.cat([f1_from_b1, f1_from_b2], dim=1) # Concatenate along channel dim -> [B, 2C, H, W]
        f1_final = self.fusion_conv(f1_cat)

        # 4. Fuse the outputs for the second temporal feature (f2)
        f2_cat = torch.cat([f2_from_b1, f2_from_b2], dim=1) # Concatenate along channel dim -> [B, 2C, H, W]
        f2_final = self.fusion_conv(f2_cat)

        return f1_final, f2_final