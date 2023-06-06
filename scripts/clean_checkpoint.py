import torch

state_dict = torch.load('./outputs/stackflow3/intercap/_latest_checkpoint.pth')
new_state_dict = {
    'epoch': state_dict['epoch'],
    'backbone': state_dict['backbone'],
    'header': state_dict['header'],
    'stackflow': state_dict['stackflow'],
}
torch.save(new_state_dict, './outputs/stackflow3/intercap/latest_checkpoint.pth')
