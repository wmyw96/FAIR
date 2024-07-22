import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sympy import GramSchmidt,Matrix




def run_irm(pca_comp):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=1)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=100.0)
    parser.add_argument('--steps', type=int, default=20001)
    parser.add_argument('--grayscale_model', action='store_true')
    parser.add_argument( '--method', choices=['LASSO','FAIR','GroupDRO','IRM'])
    flags = parser.parse_args()

    print('Flags:')
    for k,v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))
    final_train_accs = []
    final_test_accs = []


    # Load MNIST, make train/val splits, and shuffle train set examples
    # Build environments

    def make_environment(r1,r2,mode,begin,end):
        im = np.load(f'./res/{mode}/rwater_{r1}_rland_{r2}_x.npy')[begin:end]
        im = torch.from_numpy(np.dot(im-np.mean(im,axis=0),pca_comp.T)).to(torch.float32)
        y =  torch.from_numpy(np.load(f'./res/{mode}/rwater_{r1}_rland_{r2}_y.npy')[begin:end]).view(-1,1)
        return {
        'images': im.cuda(),
        'labels': y.cuda()
        }

    envs = [
        make_environment(0.95,0.9,'train',0,50000),
        make_environment(0.75,0.7,'train',0,50000),
        make_environment(0.02,0.02,'test',0,30000)
    ]

    # Define and instantiate the model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(500, flags.hidden_dim)
            else:
                lin1 = nn.Linear(500, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 1)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = lin1
        def forward(self, input):
            out = self._main(input)
            return out

    mlp = MLP().cuda()

    # Define loss function helpers

    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-1).float().mean()

    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    # Train loop

    def pretty_print(*values):
        col_width = 13
        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)
        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight 
            if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            train_penalty.detach().cpu().numpy(),
            test_acc.detach().cpu().numpy()
        )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
