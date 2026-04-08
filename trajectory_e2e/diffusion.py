import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from src_obj.common import *
import pdb
import math
from tqdm import tqdm


def mask_mse_func(input1, input2, mask):
    fn = torch.nn.MSELoss(reduction='mean')
    return fn(input1[mask==1], input2[mask==1])

class VarianceSchedule(Module):

    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)
        posterior_mean_coef1 = torch.zeros_like(sigmas_flex)
        posterior_mean_coef2 = torch.zeros_like(sigmas_flex)
        for i in range(1, posterior_mean_coef1.size(0)):
            posterior_mean_coef1[i] = torch.sqrt(alpha_bars[i-1]) * betas[i] / (1 - alpha_bars[i])
        posterior_mean_coef1[0] = 1.0
        for i in range(1, posterior_mean_coef2.size(0)):
            posterior_mean_coef2[i] = torch.sqrt(alphas[i]) * (1 - alpha_bars[i-1]) / (1 - alpha_bars[i])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    # def denoise_fn(self, x_0, curr=None, context=None, timestep=0.08, t=None, mask=None):
    def denoise_fn(self, x_0, context=None, timestep=0.08, t=None, mask=None):

        batch_size = x_0.shape[0]
        # point_dim = x_0.shape[-1]
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)
        
        

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)******T
        x_t = c0 * x_0 + c1 * e_rand
        # nei_list = None
        # p0 = curr[...,:2] # B, N, 2

        # if self.config.nei_padding_mask==True:
        #     nei_list = self.curr_padding(p0)

        x_0_hat = self.net(x_t, beta=beta, context=context)

        # x_0_hat = self.net(x_t, beta=beta, context=context, nei_list = nei_list, t=t)
        return x_0_hat
    
    def get_loss(self, x_0, context=None, timestep=0.08, t=None, mask=None,traj_rand=None,pt_num=50):

        # batch_size = x_0.shape[0]
        # point_dim = x_0.shape[-1]
        x_0_hat = self.denoise_fn(x_0, context=context, timestep=timestep, t=t, mask=mask)

        if mask==None:
            x_0_hat = x_0_hat.reshape(-1,2,pt_num,2)
            x_0 = x_0.reshape(-1,2,pt_num,2)
            batch_size = x_0.size(0)
            if traj_rand is not None:
                traj_rand = traj_rand.reshape(-1,2,1)
                mask = traj_rand==0
                mask = mask.expand(batch_size,2,pt_num*2).reshape(batch_size,-1)
                loss = F.mse_loss(x_0_hat.view(batch_size,-1).float() * mask, 
                                  x_0.view(batch_size,-1).float()* mask, reduction='sum') / (mask.sum() + 1e-5)
            else:
                loss = F.mse_loss(x_0_hat.float(), x_0.float(), reduction='mean')
        else:
            loss = mask_mse_func(x_0_hat, x_0, mask)       
        return loss,x_0_hat.unsqueeze(0)
    

    def sample(self, num_points,context,sample,bestof,point_dim=2,flexibility=0.0, ret_traj=False,sampling="ddpm", step=100):
        """_summary_

        Args:
            num_points (_type_): _description_
            context (_type_): [bs(sample), obs_len+1(destination),N,2]
            sample (_type_): _description_
            bestof (_type_): _description_
            point_dim (int, optional): _description_. Defaults to 2.
            flexibility (float, optional): _description_. Defaults to 0.0.
            ret_traj (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # point_dim=2
        traj_list = []
        # history = context[0]
        # ped_features = context[1]
        # assert context[0].shape[0]==context[1].shape[0]
        # if curr!=None:
        #     assert len(curr.shape)==3 # t, n, 6
        batch_size = context.shape[0]
        # num_ped = context[1].shape[1]
        if bestof:
            # x_T = torch.randn([batch_size, num_points, point_dim]).to('cuda')
            x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)

        else:
            # x_T = torch.zeros([batch_size, num_points, point_dim]).to('cuda')
            x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)

        traj = {self.var_sched.num_steps: x_T}
        pbar = range(self.var_sched.num_steps, 0, -1)
        # nei_list = None
        # p0 = curr[...,:2] # B, N, 2

        # if self.config.nei_padding_mask==True:
        #     nei_list = self.curr_padding(p0)


        for t in tqdm(pbar,f"setp are {len(pbar)}, DDPM sampling ..."):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha_bar_ = self.var_sched.alpha_bars[t-1]

            c0 = torch.sqrt(alpha_bar_).view(-1, 1, 1).cuda()       # (B, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar_).view(-1, 1, 1).cuda()

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            

            x_0_hat = self.net(x_t, beta=beta, context=context)
            mean, var = self.p_mean_variance(x_0_hat,x_t,t)
            x_next = mean + torch.sqrt(var)*z
            assert x_next.shape == x_t.shape
            traj[t-1] = x_next
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            

        if ret_traj:
            traj_list.append(traj)
        else:
            traj_list.append(traj[0])
        
        return torch.stack(traj_list),None
    # 11111111111

    def sample_ddim(self, num_points, context, sample, bestof, point_dim=2, flexibility=0.0, ret_traj=False, sampling="ddim", step=100):
        sqrt_alphas_bar = torch.sqrt(self.var_sched.alpha_bars)
        sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.var_sched.alpha_bars)
        
        traj_list = []
        for i in range(sample):
            batch_size = context.size(0)
            if bestof:
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
            traj = {self.var_sched.num_steps: x_T}
            stride = step
            # stride = int(100/stride)
            for t in tqdm(range(self.var_sched.num_steps, 0, -stride),f"setp are {str(step)}, DDIM sampling ..."):
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t-stride]
                #pdb.set_trace()
                sigma = self.var_sched.get_sigmas(t, flexibility)

                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                x_t = traj[t]
                beta = self.var_sched.betas[[t]*batch_size]
                x_0 = self.net(x_t, beta=beta, context=context)
                e_theta = self.predict_eps_from_x(x_t, x_0, torch.tensor([t]).to(x_t.device),
                                                sqrt_alphas_bar,sqrt_one_minus_alphas_bar)

                # e_theta = self.net(x_t, beta=beta, context=context)
                if sampling == "ddpm":
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    pdb.set_trace()
                traj[t-stride] = x_next.detach()     # Stop gradient and save trajectory.
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.

            if ret_traj:
                traj_list.append(traj)
            else:
                traj_list.append(traj[0])
                
        return torch.stack(traj_list),None

    

    def predict_eps_from_x(self, z_t, x_0, t,sqrt_alphas_bar,sqrt_one_minus_alphas_bar):
        
        
        eps = (
            (z_t - self.extract(sqrt_alphas_bar, t, x_0.shape) * x_0) / 
            self.extract(sqrt_one_minus_alphas_bar, t, x_0.shape))
        return eps


    def extract(self, v, t, x_shape):
        """
        Extract some coefficients at specified timesteps, then reshape to
        [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        out = torch.gather(v, index=t, dim=0).float()
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


    def p_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self.var_sched.posterior_mean_coef1[t] * x_start +
            self.var_sched.posterior_mean_coef2[t]  * x_t
        )
        posterior_variance = self.var_sched.sigmas_inflex[t]#.view(x_start.shape[0],*[1]*(x_start.ndim-1))
        
        # assert (posterior_mean.shape[0] == posterior_variance.shape[0]  ==
        #         x_start.shape[0])
        return posterior_mean, posterior_variance



        
    
    
    

class TransformerConcatLinear(Module):

    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1)
        self.concat1 = ConcatSquashLinear(2,2*context_dim,context_dim+3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, 2, context_dim+3)
        #self.linear = nn.Linear(128,2)


    def forward(self, x, beta, context): #torch.Size([32, 50, 2])  32  torch.Size([32, 1, 224])
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        x = self.concat1(ctx_emb,x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)


        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)


