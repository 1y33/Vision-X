import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class show:
    @staticmethod
    def show_from_tensor(tensor: torch.tensor, label=None, cmap=None,plt_figure=(5,5)):
        if cmap is None:
           if (tensor.shape[0] == 1):
                cmap = "gray"
           else :
                cmap ="RdYlBu"

        image = tensor.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))

        if label is not None:
            plt.title(label)

        plt.figure(figsize=plt_figure)
        plt.imshow(image,cmap=cmap)
        plt.axis(False)
        plt.show()

    @staticmethod
    def show_grid(dataset,num=25):
        rnd_indx = np.random.randint(0,len(dataset),num)
        x_grid = [dataset[i][0] for i in rnd_indx]
        y_grid = [dataset[i][1] for i in rnd_indx]

        grid = torchvision.utils.make_grid(x_grid,padding=1,nrow=round(np.sqrt(num)))


        show.show_from_tensor(grid,str(y_grid))

    @staticmethod
    def show_batch(tensor,num=25):
        data = tensor.detach().cpu()
        grid = torchvision.utils.make_grid(data[:num], nrow=5).permute(1, 2, 0).numpy()
        plt.imshow(grid.clip(0, 1))
        plt.show()

class convolution_calculation:

    @staticmethod
    def conv2d_hw_calculation(h,w,conv,pool=2):
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation

        h_out = (h + 2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1
        w_out = (w + 2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1] + 1

        h_out = np.floor(h_out)
        w_out = np.floor(w_out)

        if pool:
            h_out /=pool
            w_out /=pool

        return int(h_out), int(w_out)

    @staticmethod
    def convtranspose2dd(h,w,convtr):
        stride = convtr.stride
        padding = convtr.padding
        dilation = convtr.dilation
        kernel_size = convtr.kernel_size
        output_padding = convtr.output_padding

        h_out = (h - 1)*stride[0] - 2*padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        w_out = (w - 1)*stride[1] - 2*padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[0] + 1

        return int(h_out),int(w_out)

    