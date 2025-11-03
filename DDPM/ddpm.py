import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from modelnet.TTCM import Net
from torch.nn import functional as F
from metrics_utils import *


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            image_size,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            device=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model.to(device)
        self.loss_type = loss_type
        self.conditional = conditional
        self.device = device
        self.set_new_noise_schedule()

    def set_loss(self):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(self.device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(self.device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        betas = make_beta_schedule(
            schedule="linear",
            n_timestep=2000,
            linear_start=1e-4,
            linear_end=2e-2)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_recon, x_t, t):
        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_recon, t, x_t):
        model_mean, model_log_variance = self.p_mean_variance(x_recon=x_recon, t=t, x_t=x_t)
        noise = torch.randn_like(x_recon) if t > 0 else torch.zeros_like(x_recon)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, lrhs, hrMS, gt, noise=None):
        b, c, h, w = gt.shape
        print(b,c,h,w)
        self.model = self.model.to(self.device)
        img = torch.randn([b, c, h, w]).to(self.device)
        pn = np.array([])
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                      total=self.num_timesteps):
            x_recon = self.model(F.interpolate(lrhs, scale_factor=4, mode='bicubic'), hrMS,
                                    torch.cat([F.interpolate(lrhs, scale_factor=4, mode='bicubic'), img], dim=1),
                                    noise_level=torch.FloatTensor(
                                        np.repeat(self.sqrt_alphas_cumprod_prev[i], b)).view(b, 1).to(self.device))
            img = self.p_sample(x_recon=x_recon, x_t=img, t=i)
        img = self.model(F.interpolate(lrhs, scale_factor=4, mode='bicubic'), hrMS, torch.cat([x_recon, img], dim=1),
                         torch.zeros([b, 1]).to(self.device))
        print(pn)
        return img

    # @torch.no_grad()
    # def sample(self, batch_size=1, continous=False):
    #     image_size = self.image_size
    #     channels = self.channels
    #     return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, lrhs, hrMS, gt, noise=None):
        return self.p_sample_loop(lrhs=lrhs, hrMS=hrMS, gt=gt, noise=noise)

    @torch.no_grad()
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )

    def p_losses(self, lrhs, hrms, gt, noise=None):
        x_start = gt
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps)
        noise = torch.randn([b, c, h, w]).to(self.device)
        # from x0 to x_t
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.repeat(self.sqrt_alphas_cumprod_prev[t], b)).to(self.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
        # x_0_t = self.predict_start_from_noise(x_noisy, t, noise=noise)
        # from x0 to x_(t-1)
        # continuous_sqrt_alpha_cumprod1 = torch.FloatTensor(
        #     np.repeat(self.sqrt_alphas_cumprod_prev[t - 1], b)).to(self.device)
        # continuous_sqrt_alpha_cumprod1 = continuous_sqrt_alpha_cumprod1.view(b, -1)
        # x_noisy1 = self.q_sample(
        #     x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod1.view(-1, 1, 1, 1),
        #     noise=noise)
        lrhs = F.interpolate(lrhs, scale_factor=4, mode='bicubic')
        # pred x0
        x_recon = self.model(
            lrhs, hrms, torch.cat([lrhs, x_noisy], dim=1),
            continuous_sqrt_alpha_cumprod)

        # pred_x = self.p_sample(x_recon, t, x_noisy)  # predx = x_(t-1)
        loss1 = self.loss_func(x_recon, gt)
        # loss2 = self.loss_func(pred_x, )x_noisy1
        return loss1  # loss2

    def forward(self, lrhs, hrms, gt, noise=None):
        return self.p_losses(lrhs, hrms, gt, noise=noise)


class DDPM:
    def __init__(self, device, image_size=160, channels=102, loss_type='l1', conditional=True, load_net=False):
        super(DDPM, self).__init__()
        # define network and load pretrained models
        self.device = device
        print(device)
        model = Net(num_channels=channels, device=device).to(device)
        self.netG = self.set_device(GaussianDiffusion(model,
                                                      image_size=image_size,
                                                      channels=channels,
                                                      loss_type=loss_type,
                                                      conditional=conditional, device=device))
        self.schedule_phase = None
        self.noise = None
        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule()
        self.netG.train()
        # find the parameters to optimize
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=0.00001)
        if load_net:
            self.load_network()

    def set_device(self, var):
        var = var.to(self.device)
        return var

    def optimize_parameters(self, lrhs, hrms, gt):
        self.netG.to(self.device)
        self.optG.zero_grad()

        loss1 = self.netG(lrhs, hrms, gt, noise=self.noise)
        # need to average in multi-gpu
        # b, c, h, w = gt.shape
        loss1.backward()
        self.optG.step()
        return loss1

    def test(self, lrhs, hrms, gt):
        self.netG.eval()
        with torch.no_grad():
            self.SR = self.netG.super_resolution(
                lrhs, hrms, gt, noise=self.noise)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        self.netG.set_loss()

    def set_new_noise_schedule(self):
        self.netG.set_new_noise_schedule()

    def get_current_visuals(self, need_LR=True, sample=False):
        pass

    def print_network(self):
        pass

    def save_network(self, epoch=False):

        network = self.netG
        network = network.to('cpu')
        state_dict = {'modelnet': network.state_dict()}
        if epoch:
            torch.save(state_dict, 'pavia_last_epoch_{}.pt'.format(epoch))
        else:
            torch.save(state_dict, 'pavia_last_best.pt')

    def load_network(self):
        state = torch.load('pavia_last_epoch_150.pt')
        self.netG.load_state_dict(state['modelnet'])
        self.netG = self.set_device(self.netG)
        print('load finish')
