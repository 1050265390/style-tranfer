import torch
import torch.optim as optim                      # 导入 pytorch 优化模块
import torch.nn.functional as F                  # 导入函数包

from PIL import Image                            # 用于加载和显示图片
import matplotlib.pyplot as plt

from torchvision import transforms               # 对图像数据进行处理
from torchvision import models

import copy

# 内容损失
class ContentLoss(torch.nn.Module):                  # 构造模型类，继承自Module,构造出来的对象会根据计算图主动实现backward()过程

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # 动态计算梯度
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)            # 损失函数
        return input

# 风格损失
def gram_matrix(input):
    a, b, c, d = input.size()
    # a=1,是batch_size
    # b为特征图数量
    # (c, d)为f的维度(N=c*d)
    features = input.view(a * b, c * d)

    G = torch.matmul(features, features.t())
    # 每个特征图除以元素数量来归范化gram矩阵
    return G.div(a * b * c * d)
class StyleLoss(torch.nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features           # 导入预训练好的神经网络。使用19层的VGG网络，把所有训练好的权重和参数加载到cnn里面

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
# 标准化操作，让数据更有效地传递，避免用激活函数后不能输出有效数据
# 创建一个模块来标准化输入图像，以便我们可以轻松地将其放入  nn.Sequential
class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # 查看均值和标准差以使其为[C x 1 x 1]，以便它们可以直接使用形状为[B x C x H x W]的图像张量。
        # B是批量大小。 C是通道数。 H是高度，W是宽度。
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)

        self.mean = mean.clone().detach().view(-1, 1, 1)          # 返回新的tensor，不用梯度
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# 所需的深度层以计算样式/内容损失：
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # 标准化模块
    normalization = Normalization(normalization_mean, normalization_std)
    # 只是为了获得对内容/样式的可迭代访问或列表
    # losses
    content_losses = []
    style_losses = []

    # 假设cnn是nn.Sequential，那么我们创建一个新的nn.Sequential
    # 放入应该顺序激活的模块
    model = torch.nn.Sequential(normalization)
    i = 0  # 每当转换时就增加
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):                       # isinstance用于判断对象是否是一个已知的类型
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, torch.nn.ReLU):
            name = 'relu_{}'.format(i)
            # 旧版本与我们在下面插入的ContentLoss和StyleLoss不能很好地配合使用。
            # 因此，我们在这里替换为不适当的。
            layer = torch.nn.ReLU(inplace=False)
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # 增加内容损失:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # 增加风格损失:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 现在我们在最后一次内容和样式丢失后修剪图层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps,
                       style_weight=1000000, content_weight=1, s_data=[], c_data=[], x_data=[], t_data=[]):

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)

    optimizer = optim.LBFGS([input_img])                    #优化器函数，对图片进行训练，优化，更新，传入input_img的所有参数，默认学习率lr为1e-8
    print('start train..')
    run = [0]
    while run[0] <= num_steps:             # 循环训练次数

        def closure():
            # 更正更新后的输入图像的值
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()          # 做反向传播之前把梯度归零,因为每次计算loss更新以后梯度都会保留在模型里面。
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight                   # 风格损失和内容损失都乗上各自的超参数权重
            content_score *= content_weight

            loss = style_score + content_score             # 总损失等于风格损失加上内容损失
            loss.backward()                                # 反向传播，计算梯度，不保留计算图，给每一个神经网络中的结点计算出梯度

            run[0] += 1
            if run[0] % 5 == 0:
                print(run[0])
                s_data.append(style_score.item())
                c_data.append(content_score.item())        # 将训练中的损失的标量值加入列表中.
                t_data.append(loss.item())
                x_data.append(run[0])
            return style_score + content_score

        optimizer.step(closure)                            # 进行更新，优化梯度
    input_img.data.clamp_(0, 1)                            # 更新图像数据
    return input_img


unloader = transforms.ToPILImage()  # 转换为PIL图像
plt.ion()                           # 交互模式，开启多个窗口


def imshow(tensor, title=None):
    image = tensor.clone()           # 我们克隆张量不对其进行更改
    image = image.squeeze(0)         # 还原到原来的维度
    image = unloader(image)          # Tensor转化为PIL图片
    plt.imshow(image)                # 将矩阵数据显示成图片。
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 稍停一下，以便更新地块，实现动态绘图


if __name__ == '__main__':
    # 预处理，将图像(PIL)转换为torch tensor
    loader = transforms.Compose([  # 把中括号里面的图像进行处理
        transforms.Resize([330, 440]),  # 缩放导入的图像,参数是图像的尺寸，让图像的尺寸一致
        transforms.ToTensor()])  # 将图像(PIL)转换为torch tensor,变成张量的维度，像素取值变为0到1


    # PIL加载图片并转化为Tensor
    def image_loader(image_name):
        image = Image.open(image_name)
        # 需要伪造的批次尺寸以适合网络的输入尺寸
        image = loader(image).unsqueeze(0)  # 升维，torch不能处理一维数据
        return image

    style_img = image_loader("./images/神奈川冲浪.jpg")
    content_img = image_loader("./images/校门口.jpg")
    assert style_img.size() == content_img.size()  # "我们需要导入相同大小的样式和内容图像"

    plt.figure()
    imshow(style_img, title='Style Image')
    plt.figure()
    imshow(content_img, title='Content Image')

    # input_img是tensor类型，需要把requires_grad()设置成True这样子才可以训练
    input_img = content_img.clone().requires_grad_(True)  # 克隆一张内容图用作输入图, 内容和content类似，先初始化为和content一样的内容。
    s_data = []
    c_data = []
    x_data = []
    t_data = []
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, 120,
                                1000000, 1, s_data, c_data, x_data, t_data)
    plt.figure()
    plt.plot(x_data, s_data, marker='o', mec='r', mfc='w')
    plt.plot(x_data, c_data, marker='*', ms=3)
    plt.title('content and stytle loss')
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.plot(x_data, t_data)
    plt.title('total loss')
    plt.xlabel('times')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    imshow(output, title='Output Image')

    plt.ioff()                             # 使用ioff()关闭交互模式，图像停留
    plt.show()
    # image = output.clone()  # 我们克隆张量不对其进行更改
    # image = image.squeeze(0)  # 还原到原来的维度
    # image = unloader(image)  # Tensor转化为PIL图片
    # image.save('./result.jpg')
    # plt.title('Result')
