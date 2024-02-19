import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image

import utils
from utils import show

from torch.utils.data import Dataset,DataLoader


## Generator model
class Generator(nn.Module):
# new width and height : (n-1)*stride - 2*padding +ks
  def __init__(self,z_dim,d_dim=16):
    super().__init__()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = device
    self.z_dim = z_dim
    self.gen = nn.Sequential(
        self.GenBlock(z_dim,d_dim * 64,4,1,0),
        # 200 x 1 x 1 -> 1024 x 4 x 4
        self.GenBlock(d_dim*64,d_dim*32),
        # 1024 x 4 x 4 -> 512 x 8 x 8
        self.GenBlock(d_dim*32,d_dim*16),
        # 512 x 8 x 8 -> 256 x 16 x 16
        self.GenBlock(d_dim*16,d_dim*8),
        # 256 x 16 x 16 -> 128 x 32 x 32
        self.GenBlock(d_dim*8,d_dim*4),
        # 128 x 32 x 32 -> 64 x 64 x 64
        self.GenBlock(d_dim*4,d_dim*2),
        # 64 x 64 x 64 -> 32 x 128 x 128
        self.GenBlock(d_dim*2,3),
        # 16 x 256 x 256
        nn.Tanh(), # from -1 to 1
    )

  def forward(self,noise):
    x = noise.view(len(noise),self.z_dim,1,1) # 128 x 200 x 1 x 1
    return self.gen(x)

  def GenBlock(self,in_layers, out_layers, kernel=4, stride=2, padding=1):
    # Default is for /2 sampling
    return nn.Sequential(
          nn.ConvTranspose2d(in_layers, out_layers, kernel, stride, padding),
          nn.BatchNorm2d(out_layers),
          nn.ReLU(inplace=True),)

  def gen_noise(self,num,z_dim):
    return torch.randn(num,z_dim,device=self.device)
## Critic model
class Critic(nn.Module):
  def __init__(self,d_dim=16):
    super().__init__()

    #256 x 256
    self.critic = nn.Sequential(
        self.CritBlock(3,d_dim), # 128
        self.CritBlock(d_dim,d_dim*2), # 64
        self.CritBlock(d_dim*2,d_dim*4), # 32
        self.CritBlock(d_dim*4,d_dim*8), # 16
        self.CritBlock(d_dim*8,d_dim*16), # 8
        self.CritBlock(d_dim*16,d_dim*32), # 4

        nn.Conv2d(d_dim*32,1,4,1,0) # 1
    )

  def forward(self,image):
    #128 x 3 x 256 x 256
    crit_pred = self.critic(image) #128 x 1 x 1 x1
    crit_pred = crit_pred.view(len(crit_pred),-1)

    return crit_pred

  def CritBlock(self,in_layers, out_layers, kernel=4, stride=2, padding=1):
      return nn.Sequential(
          nn.Conv2d(in_layers, out_layers, kernel, stride, padding),
          nn.InstanceNorm2d(out_layers),
          nn.LeakyReLU(0.2),
      )
## Trainer class
class Trainer:

    def __init__(self,Generator,Critic,Data,Path,z_dim):
        self.Path = Path
        self.Generator = Generator
        self.Critic = Critic
        self.LR = 1e-3
        self.Generator_optimizer = torch.optim.Adam(self.Generator.parameters(),
                                                    lr= self.LR,
                                                    betas = (0.5,0.9))
        self.Critic_optimizer = torch.optim.Adam(self.Critic.parameters(),
                                                 lr = self.LR,
                                                 betas = (0.5,0.9))
        self.data = Data
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.z_dim = z_dim

    def get_gradient_penality(self,real,fake,critic,alpha,gamma=10):
        mix_images = real * alpha + (1 - alpha) * fake
        mix_scores = self.Critic(mix_images)

        gradient = torch.autograd.grad(
            inputs = mix_images,
            outputs = mix_scores,
            grad_outputs = torch.ones_like(mix_scores),
            retain_graph = True,
            create_graph = True,
        )[0]

        gradient = gradient.view(len(gradient),-1)
        gradient_norm = gradient.norm(2,dim=1)

        gradient_penality = ((gradient_norm-1)**2).mean()

        return gradient_penality

    def save_checkpoint(self):
        os.mkdir(os.path.join(self.Path,"models"))
        save_path = os.path.join(self.Path,"models","weights")
        generator_file = os.path.join(save_path,"generator.pt")
        critic_file = os.path.join(save_path,"critic.pt")

        torch.save(self.Generator.state_dict(),f=generator_file)
        torch.save(self.Critic.state_dict(),f=critic_file)

    def load_checkpoint(self):
        load_path = os.path.join(self.Path,"models","weights")
        generator_file = os.path.join(load_path,"generator.pt")
        critic_file = os.path.join(load_path,"critic.pt")

        self.Generator.load_state_dict(torch.load(generator_file))
        self.Critic.load_state_dict(torch.load(critic_file))

    def train(self,n_epochs):
        self.crit_losses = []
        self.gen_losses = []
        cur_step = 0
        best_loss = 1000
        for epoch in range(n_epochs):
            for real, _ in self.data:
                cur_bs = len(real)
                real = real.to(self.device)
                # Critic
                mean_critic_loss = 0
                for _ in range(5):
                    self.Critic_optimizer.zero_grad()

                    noise = self.Generator.gen_noise(cur_bs, self.z_dim)
                    fake = self.Generator(noise)
                    crit_fake_pred = self.Critic(fake)
                    crit_real_pred = self.Critic(real)

                    alpha = torch.rand(len(real), 1, 1, 1, device=self.device, requires_grad=True)
                    gp = self.get_gradient_penality(real, fake.detach(), self.Critic, alpha)
                    crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp

                    mean_critic_loss += crit_loss.item() / 5

                    crit_loss.backward()
                    self.Critic_optimizer.step()

                    self.crit_losses += [mean_critic_loss]

                # Generator
                self.Generator_optimizer.zero_grad()
                noise = self.Generator.gen_noise(cur_bs, self.z_dim)
                fake = self.Generator(noise)
                crit_fake_pred = self.Critic(fake)

                gen_loss = -crit_fake_pred.mean()
                gen_loss.backward()
                self.Generator_optimizer.step()

                self.gen_losses += [gen_loss.item()]

                # Stats
                if (cur_step % 10 == 0 and cur_step > 0):
                    utils.show.show_batch(fake)
                    utils.show.show_batch(real)

                    print(f"Epoch:{epoch + 1} | Step: {cur_step + 1}\r")
                    print(f"Gen Loss: {gen_loss} | Crit Loss: {crit_loss}")

                cur_step += 1

            if (gen_loss < best_loss):
                best_loss = gen_loss
                self.save_checkpoint()

            self.load_checkpoint()

class custom_dataset(Dataset):
    def __inti__(self,path):
        self.path = path
        self.path_list = os.listdir(self.path)
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self,idx):
        img_name = os.path.join(self.path,self.path_list[idx])
        img = Image.open(img_name)
        img = self.transform(img)
        _ = 0 # random_value will remove it in the futre but i am lazy
        return img , _

def create_dataloader(path_2_data_folder):
    ds = custom_dataset(path_2_data_folder)
    dl = DataLoader(ds,batch_size =32 ,shuffle=True)

    return dl

def train(path_2_data_folder,data=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator_model = Generator(z_dim=200).to(device)
    critic_model = Critic().to(device)
    if data is None:
         data = create_dataloader(path_2_data_folder)
    else:
        data = data
    training = Trainer(Generator=generator_model,
                       Critic=critic_model,
                       Data= data,
                       Path=path_2_data_folder,
                       z_dim=200)

    training.train(1000)
    return generator_model,critic_model


def generate_image(generator,z_dim=200):
    img = generator.gen_noise(z_dim=200)
    show.show_from_tensor(img)





