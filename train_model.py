from SCINet import *

def train(data, label):
    model = SCINet(output_len=2, input_len=30, input_dim=60, hid_size=1,
                   num_stacks=2,
                   num_levels=2, concat_len=0, groups=1, kernel=3, dropout=0.5,
                   single_step_output_One=0, positionalE=True,
                   modified=True).cuda()
    data = torch.FloatTensor(data).cuda()
    y = model(data)
    return y