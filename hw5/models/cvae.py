import torch
import torch.nn as nn
from models.vgg_64 import vgg_encoder, vgg_decoder
from models.lstm import gaussian_lstm, lstm
from utils import init_weights, mse_metric, kl_criterion
import random

class cvae(nn.Module):
    def __init__(self, args):
        super(cvae, self).__init__()
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.encoder = vgg_encoder(args.g_dim)
        self.posterior = gaussian_lstm(args.g_dim, args.z_dim, args.rnn_size, 
            args.posterior_rnn_layers, args.batch_size, device)
        self.frame_predictor = lstm(args.g_dim + args.z_dim + args.cond_dim, args.g_dim, 
            args.rnn_size, args.predictor_rnn_layers, args.batch_size, device)
        self.decoder = vgg_decoder(args.g_dim)
        self.args = args
        self.apply(init_weights)
    def forward(self, x, cond, tfr):
        ## Initialize the hidden state.
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()

        mse = 0
        kld = 0
        mse_criterion = nn.MSELoss()
        use_teacher_forcing = True if random.random() < tfr else False
        h_seq = [self.encoder(x[i]) for i in range(self.args.n_past+self.args.n_future)]

        for t in range(1, self.args.n_past + self.args.n_future):
            ## Encode the image at step (t-1)
            if self.args.last_frame_skip or (t < self.args.n_past): # origin: t <= self.args.n_past
                h_in, skip = h_seq[t-1] # self.encoder(x[t-1])
            else:
                if use_teacher_forcing:
                    h_in, _ = h_seq[t-1] # self.encoder(x[t-1])
                else:
                    h_in, _ = self.encoder(x_pred)

            ## Obtain latent vector z at step (t)
            h_t, _ = h_seq[t] # self.encoder(x[t])
            z_t, mu, logvar = self.posterior(h_t)

            ## Decode the image
            lstm_in = torch.cat([cond[t-1], h_in, z_t], dim=1) # origin [h_in, z_t, cond[t-1]]
            g_t = self.frame_predictor(lstm_in)
            x_pred  = self.decoder([g_t, skip])

            # calculate loss
            mse += mse_criterion(x_pred, x[t])
            kld += kl_criterion(mu, logvar, self.args)

        return mse, kld

    def predict(self, x, cond):
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()

        pred_seq = []
        pred_seq.append(x[0])
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        mse = 0
        kld = 0
        mse_criterion = nn.MSELoss()
        h_seq = [self.encoder(x[i]) for i in range(self.args.n_past+self.args.n_future)]

        for t in range(1, self.args.n_past + self.args.n_future):
            # Encode the image at step (t-1)
            if self.args.last_frame_skip or (t < self.args.n_past): # origin: t <= self.args.n_past
                h_in, skip = h_seq[t-1] # self.encoder(x[t-1])
            elif t==self.args.n_past:
                h_in, _ = h_seq[t-1]
            else:
                h_in, _ = self.encoder(x_pred)

            ## Obtain latent vector z at step (t)
            if t < self.args.n_past:
                h_t, _ = h_seq[t] # self.encoder(x[t])
                z_t, mu, logvar = self.posterior(h_t)
            else:
                z_t = torch.randn(self.args.batch_size, self.args.z_dim).to(device)

            ## Decode the image
            lstm_in = torch.cat([cond[t-1], h_in, z_t], dim=1) # origin [h_in, z_t, cond[t-1]]
            g_t = self.frame_predictor(lstm_in)
            x_pred  = self.decoder([g_t, skip])
            
            ## calculate loss
            mse += mse_criterion(x_pred, x[t])
            kld += kl_criterion(mu, logvar, self.args)

            if t < self.args.n_past:
                pred_seq.append(x[t])
            else:
                pred_seq.append(x_pred)

        pred_seq = torch.stack(pred_seq)

        return pred_seq , mse, kld