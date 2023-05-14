from tqdm import tqdm
import torch


def joint_optimize(hoi_instance, loss_functions, loss_weights, iterations=3, steps_per_iter=200, lr=0.01):

    optimizer = hoi_instance.get_optimizer(lr=lr)
    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        for i in loop:
            optimizer.zero_grad()
            hoi_dict = hoi_instance()
            losses = {}
            for f in loss_functions:
                losses.update(f(hoi_dict))
            loss_list = [loss_weights[k](v, it) for k, v in losses.items()]
            total_loss = torch.stack(loss_list).sum()

            total_loss.backward()
            optimizer.step()

            l_str = 'Iter: {}'.format(i)
            for k, v in losses.items():
                l_str += ', {}: {:0.4f}'.format(k, v.detach().item())
                loop.set_description(l_str)
