import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16

from einops import rearrange, repeat

from tab_transformer_pytorch import TabTransformer

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# define NN for propensity score

class PS(nn.Module):

    def __init__(self, in_N, m, depth=2):
        super(PS, self).__init__()

        self.stack = nn.ModuleList()
        self.stack.append(nn.Linear(in_N, m))

        for i in range(depth - 1):
            self.stack.append(nn.Linear(m, m))

        self.stack.append(nn.Linear(m, 1))
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.stack[0](x))
        x = self.dropout(x)
        for i in range(1, len(self.stack) - 1):
            x = self.act(self.stack[i](x))
            x = self.dropout(x)
        x = self.stack[-1](x)
        x = 1e-2 + (0.98) * self.sigmoid(x)  # 0.01-0.99
        return x


class PS_TransformerDebias(nn.Module):

    def __init__(
            self,
            model_config,
            depth=6,
            heads=8
    ):
        super(PS_TransformerDebias, self).__init__()

        self.model = TabTransformer(
            categories=model_config["categories"],  # tuple containing the number of unique values within each category
            num_continuous=model_config["num_continuous"],  # number of continuous values
            dim=32,  # dimension, paper set at 32
            dim_out=model_config["feat_dim"],  # binary prediction, but could be anything
            depth=depth,  # depth, paper recommended 6
            heads=heads,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=0.1,  # feed forward dropout
            mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            # continuous_mean_std=CONT_MEAN_STD  # (optional) - normalize the continuous values before layer norm
        )

        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()

        self.out_layer = nn.Linear(model_config["feat_dim"], 1)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_cat, x_con):

        x_cat = x_cat.long()

        x = self.model(x_cat, x_con)

        x = (x - x.mean(0)) / x.std(0)

        # x = 1e-2 + (0.98) * self.sigmoid(x)  # 0.01-0.99
        return x

    def forward_debias(self, x_cat, x_con, bias_vectors, alphas):
        """
        Forward propagation with debiasing.

        Parameters:
          x: Input tensor.
          bias_vectors: Dictionary mapping layer indices to bias vectors.
          alphas: Dictionary mapping layer indices to debiasing coefficients.

        For each layer, if the layer index is in bias_vectors, subtract
        (alpha * bias_vector) from the pre-activation output.
        """
        # Layer 0
        # a = self.stack[0](x)
        # if 0 in bias_vectors:
        #     a = a - alphas.get(0, 1.0) * bias_vectors[0]
        # x = self.act(a)
        # x = self.dropout(x)
        # # Intermediate layers
        # for i in range(1, len(self.stack) - 1):
        #     a = self.stack[i](x)
        #     if i in bias_vectors:
        #         a = a - alphas.get(i, 1.0) * bias_vectors[i]
        #     x = self.act(a)
        #     x = self.dropout(x)
        # # Output layer (no debiasing)
        # x = self.stack[-1](x)
        # x = 1e-2 + 0.98 * self.sigmoid(x)
        a = self.model(x_cat, x_con)
        a = a - alphas.get(0, 1.0) * bias_vectors[0]
        x = self.act(a)
        x = self.dropout(x)

        x = self.out_layer(x)
        x = 1e-2 + 0.98 * self.sigmoid(x)

        return x

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

class PS_Transformer(nn.Module):

    def __init__(
            self,
            model_config,
            depth=6,
            heads=8
    ):
        super(PS_Transformer, self).__init__()

        self.model = TabTransformer(
            categories=model_config["categories"],
            # tuple containing the number of unique values within each category
            num_continuous=model_config["num_continuous"],  # number of continuous values
            dim=32,  # dimension, paper set at 32
            dim_out=model_config["feat_dim"],  # binary prediction, but could be anything
            depth=depth,  # depth, paper recommended 6
            heads=heads,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=0.1,  # feed forward dropout
            mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            # continuous_mean_std=CONT_MEAN_STD  # (optional) - normalize the continuous values before layer norm
        )

        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_cat, x_con):
        x_cat = x_cat.long()

        x = self.model(x_cat, x_con)

        x = (x - x.mean(0)) / x.std(0)

        # x = 1e-2 + (0.98) * self.sigmoid(x)  # 0.01-0.99
        return x

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)


    # def pre_act(self, x):
    #     xi = self.stack[0](x)
    #     pre_act_list = [xi]
    #     x = self.act(xi)
    #     x = self.dropout(x)
    #
    #     for i in range(1, len(self.stack) - 1):
    #         xi = self.stack[i](x)
    #         pre_act_list.append(xi)
    #         x = self.act(xi)
    #         x = self.dropout(x)
    #
    #     x = self.stack[-1](x)
    #     x = 1e-2 + (0.98) * self.sigmoid(x)
    #
    #     return x, pre_act_list


