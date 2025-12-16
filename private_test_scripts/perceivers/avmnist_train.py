import torch.utils.checkpoint
torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)
import sys
import os
sys.path.insert(1,os.getcwd())
#from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from private_test_scripts.perceivers.crossattnperceiver import MultiModalityPerceiver, InputModality

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from datasets.avmnist.get_data import get_dataloader
trains4,valid4,test4=get_dataloader('datasets/avmnist/_MFAS/avmnist',no_robust=True,unsqueeze_channel=False,to4by4=True,fracs=1)
device='cuda:0'

colorless_image_modality=InputModality(
    name='colorlessimage',
    input_channels=16,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)

audio_spec_modality=InputModality(
    name='audiospec',
    input_channels=256,
    input_axis=2,
    num_freq_bands=6,
    max_freq=1
)

for i in range(1):
    #"""
    model = MultiModalityPerceiver(
        modalities=(colorless_image_modality,audio_spec_modality),
        depth=1,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
        num_latents=20,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=64,  # latent dimension
        cross_heads=1,  # number of heads for cross attention. paper said 1
        latent_heads=6,  # number of heads for latent self attention, 8
        cross_dim_head=64,
        latent_dim_head=64,
        num_classes=1,  # output number of classes
        attn_dropout=0.,
        ff_dropout=0.,
        #embed=True,
        weight_tie_layers=True,
        num_latent_blocks_per_layer=1, # Note that this parameter is 1 in the original Lucidrain implementation,
        cross_depth=1
    ).to(device)
    model.to_logitslist=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(128),torch.nn.Linear(128,10))]).to(device)

    from private_test_scripts.perceivers.train_structure_multitask import train

    os.environ['Mode'] = 'Attack'
    os.environ['AT_Methods'] = "VARMAT"
    os.environ['adv_iter'] = "10"
    os.environ['eps'] = "0.01"
    os.environ['temperature'] = "0.5"
    os.environ['method'] = "FGSM-RS"
    ckpt = 'ckpt/at_ckpt/FGSM-RS/avmnist_reg/VARMAT.pt'


    print(os.environ['adv_iter'])
    print(os.environ['eps'])

    epochs = 0

    if epochs > 0 and os.environ['adv_iter'] != "10" and os.environ['eps'] != "0.01":
        exit()

    records=train(model,epochs,[trains4],[valid4],[test4],[['colorlessimage','audiospec']],ckpt,lr=0.0008,device=device,train_weights=[1.0],is_affect=[False],unsqueezing=[False],transpose=[False],evalweights=[1],start_from=0,weight_decay=0.001)
    print(ckpt)