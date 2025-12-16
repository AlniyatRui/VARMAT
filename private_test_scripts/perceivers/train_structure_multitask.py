import torch
from eval_scripts.performance import f1_score
from tqdm import tqdm
from copy import deepcopy
import os 
import torch.nn.functional as F
import torch.utils.checkpoint
torch.utils.checkpoint.checkpoint = lambda func, *args, **kwargs: func(*args, **kwargs)
def FGSM_PCO(model, latent_out, iters=1, j=None, loss_function=None, device='cuda', init_delta=None):

    gamma = 2

    ep_grad = []

    eps_ratio = eval(os.environ['eps'])
    at = os.environ['AT_Methods']
    if os.environ['Train'] == "False":
        at = "Avg"
    B = latent_out[0].size(0)
    K = len(latent_out)

    eps_list = []
    for latent_rep in latent_out:
        # import pdb; pdb.set_trace()
        f_norm = torch.norm(latent_rep.view(B, -1), p='fro', dim=1, keepdim=True)  # shape: [B, 1]
        eps_list.append(eps_ratio * f_norm)

    adv_out = []
    t_out = []
    for m, latent in enumerate(latent_out):
        adv_out.append(latent.detach().clone())
    
    test = False
    if not torch.is_grad_enabled():
        test = True
    if test:
        torch.set_grad_enabled(True)

    if at == "Avg":
        W = [1 / K for _ in range(K)]
        Vulnerability = [torch.Tensor([1 / K]) for _ in range(K)]
        # W = [1,0,0]
    elif at == "VARMAT":
        adv_out = [latent.detach().clone().requires_grad_(True) for latent in latent_out]
        out_model = model.forward_with_latentout(latentout=adv_out)
        loss = loss_function(out_model, j[-1].to(device).long())
        
        grads = torch.autograd.grad(loss, adv_out, create_graph=True,allow_unused=True)
        loss_list = []

        for m in range(K):
            grad_flat = grads[m].reshape(adv_out[m].shape[0], -1)
            grad_norm = torch.norm(grad_flat, p=2, dim=1)                  # [B]

            per_sample_loss = grad_norm              # [B]
            loss_list.append(per_sample_loss)

        losses_tensor = torch.stack(loss_list, dim=0).transpose(0, 1).to(device)  # [B, K]

        norm_weights = losses_tensor 

        norm_weights = norm_weights.transpose(0, 1)               # [K, B]
        Vulnerability = [norm_weights[m] for m in range(K)]  

        W = [1 / K for _ in range(K)]

    if init_delta == None:
        zero_init = []
        for m, latent in enumerate(adv_out):
            delta = torch.randn_like(latent) * W[m]  * eps_list[m].view(B, 1, 1)
            delta_flat = delta.view(B, -1)
            delta_norm = torch.norm(delta_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
            eps_limit = eps_list[m] * W[m]  # [B, 1]
            scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(B, 1, 1)
            delta = delta * scale
            adv_out[m] = adv_out[m] + delta
            zero_init.append(delta.detach().clone())
    else:
        for m, latent in enumerate(adv_out):
            delta = init_delta[m].detach().clone().to(device)
            delta_flat = delta.view(B, -1)
            delta_norm = torch.norm(delta_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
            eps_limit = eps_list[m] * W[m]  # [B, 1]
            scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(B, 1, 1)
            delta = delta * scale
            adv_out[m] = adv_out[m] + delta

    for _ in range(iters):

        for m, latent in enumerate(adv_out):
            adv_out[m] = adv_out[m].detach().clone().requires_grad_()

        out_model = model.forward_with_latentout(latentout=adv_out)
        loss = loss_function(out_model, j[-1].to(device).long())
        model.zero_grad()
        loss.backward()

        for m in range(K):
            grad = adv_out[m].grad  # [B, N, D]
            grad_flat = grad.view(B, -1)
            norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-8  # [B, 1]
            norm = norm.view(B, 1, 1)

            step_size = eps_list[m].view(B, 1, 1) * W[m] / iters
            step = step_size * grad / norm
            
            ep_grad.append((grad / norm).detach().clone())

            delta = adv_out[m] + gamma * step - latent_out[m]
            delta_flat = delta.view(B, -1)
            delta_norm = torch.norm(delta_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
            eps_limit = eps_list[m] * W[m]  # [B, 1]
            scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(B, 1, 1)
            delta = delta * scale
            adv_out[m] = latent_out[m] + delta

            delta_t = adv_out[m] + step - latent_out[m]
            delta_t_flat = delta_t.view(B, -1)
            delta_t_norm = torch.norm(delta_t_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
            eps_limit = eps_list[m] * W[m]  # [B, 1]
            scale = torch.clamp(eps_limit / (delta_t_norm + 1e-8), max=1.0).view(B, 1, 1)
            delta_t = delta_t * scale
            t_out.append(latent_out[m] + delta_t)

    for m, latent in enumerate(adv_out):
        adv_out[m] = adv_out[m].detach()
        t_out[m] = t_out[m].detach()
    if test:
        torch.set_grad_enabled(False)
    # print(W)
    if init_delta == None:
        return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)], [zero_init[mod].detach().clone() for mod in range(K)], [(t_out[m] - latent_out[m]).detach().clone() for m in range(K)], Vulnerability
    return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)], [(t_out[m] - latent_out[m]).detach().clone() for m in range(K)], Vulnerability

def FGSM_EP(model, latent_out, iters=1, j=None, loss_function=None, device='cuda', init_delta=None, Weight=None):

    gamma = 1
    if os.environ['method'] == "FGSM-PCO":
        gamma = 2

    ep_grad = []

    eps_ratio = eval(os.environ['eps'])
    at = os.environ['AT_Methods']
    if os.environ['Train'] == "False":
        at = "Avg"
    # print(at)
    B = latent_out[0].size(0)
    K = len(latent_out)

    eps_list = []
    for latent_rep in latent_out:
        # import pdb; pdb.set_trace()
        f_norm = torch.norm(latent_rep.view(B, -1), p='fro', dim=1, keepdim=True)  # shape: [B, 1]
        eps_list.append(eps_ratio * f_norm)

    adv_out = []
    for m, latent in enumerate(latent_out):
        adv_out.append(latent.detach().clone())
    
    test = False
    if not torch.is_grad_enabled():
        test = True
    if test:
        torch.set_grad_enabled(True)

    if at == "Avg":
        W = [1 / K for _ in range(K)]
        Vulnerability = [torch.Tensor([1 / K]) for _ in range(K)]
        # W = [1,0,0]
    elif at == "VARMAT":
        adv_out = [latent.detach().clone().requires_grad_(True) for latent in latent_out]
        out_model = model.forward_with_latentout(latentout=adv_out)
        loss = loss_function(out_model, j[-1].to(device).long())
        
        grads = torch.autograd.grad(loss, adv_out, create_graph=True,allow_unused=True)

        loss_list = []

        for m in range(K):
            # grad_flat = adv_out[m].grad.reshape(adv_out[m].shape[0], -1)   # [B, T*D]
            grad_flat = grads[m].reshape(adv_out[m].shape[0], -1)
            grad_norm = torch.norm(grad_flat, p=2, dim=1)                  # [B]

            per_sample_loss = grad_norm                    # [B]
            loss_list.append(per_sample_loss)

        losses_tensor = torch.stack(loss_list, dim=0).transpose(0, 1).to(device)  # [B, K]

        norm_weights = losses_tensor 
        norm_weights = norm_weights.transpose(0, 1)               # [K, B]
        Vulnerability = [norm_weights[m] for m in range(K)]  

        W = [1 / K for _ in range(K)]

    for m, latent in enumerate(adv_out):
        # delta = torch.randn_like(latent) * W[m]  * eps_list[m].view(B, 1, 1)
        delta = init_delta[m].detach().clone().to(device)
        delta_flat = delta.view(B, -1)
        delta_norm = torch.norm(delta_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
        eps_limit = eps_list[m] * W[m]  # [B, 1]
        scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(B, 1, 1)
        delta = delta * scale
        adv_out[m] = adv_out[m] + delta

    for _ in range(iters):

        for m, latent in enumerate(adv_out):
            adv_out[m] = adv_out[m].detach().clone().requires_grad_()

        out_model = model.forward_with_latentout(latentout=adv_out)
        loss = loss_function(out_model, j[-1].to(device).long())
        model.zero_grad()
        loss.backward()

        for m in range(K):
            grad = adv_out[m].grad  # [B, N, D]
            grad_flat = grad.view(B, -1)
            norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-8  # [B, 1]
            norm = norm.view(B, 1, 1)


            step_size = eps_list[m].view(B, 1, 1) * W[m] / iters
            step = gamma * step_size * grad / norm
            
            ep_grad.append((grad / norm).detach().clone())

            delta = adv_out[m] + step - latent_out[m]
            delta_flat = delta.view(B, -1)
            delta_norm = torch.norm(delta_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
            eps_limit = eps_list[m] * W[m]  # [B, 1]
            scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(B, 1, 1)
            delta = delta * scale
            adv_out[m] = latent_out[m] + delta

    for m, latent in enumerate(adv_out):
        adv_out[m] = adv_out[m].detach()
    if test:
        torch.set_grad_enabled(False)
    # print(W)
    if os.environ['method'] == "FGSM-MEP":
        return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)], ep_grad, W, [eps_list[m].detach().clone() for m in range(K)], Vulnerability
    return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)], W, Vulnerability

def FGSM(model, latent_out, iters=1, j=None, loss_function=None, device='cuda'):

    ep_grad = []

    eps_ratio = eval(os.environ['eps'])
    at = os.environ['AT_Methods']
    if os.environ['Train'] == "False":
        # at = "Avg"
        eps_ratio = eps_ratio * 20
    # # print(at)
    B = latent_out[0].size(0)
    K = len(latent_out)

    eps_list = []
    for latent_rep in latent_out:
        f_norm = torch.norm(latent_rep.view(B, -1), p='fro', dim=1, keepdim=True)  # shape: [B, 1]
        eps_list.append(eps_ratio * f_norm)

    adv_out = []
    for m, latent in enumerate(latent_out):
        adv_out.append(latent.detach().clone())
    
    test = False
    if not torch.is_grad_enabled():
        test = True
    if test:
        torch.set_grad_enabled(True)

    if at == "Avg":
        W = [1 / K for _ in range(K)]
        Vulnerability = [torch.Tensor([1 / K]) for _ in range(K)]
    elif at == "VARMAT":
        adv_out = [latent.detach().clone().requires_grad_(True) for latent in latent_out]
        out_model = model.forward_with_latentout(latentout=adv_out)
        loss = loss_function(out_model, j[-1].to(device).long())
        
        grads = torch.autograd.grad(loss, adv_out, create_graph=True,allow_unused=True)

        loss_list = []

        for m in range(K):
            grad_flat = grads[m].reshape(adv_out[m].shape[0], -1)
            grad_norm = torch.norm(grad_flat, p=2, dim=1)                  # [B]
            per_sample_loss = grad_norm    # [B]
            loss_list.append(per_sample_loss)

        losses_tensor = torch.stack(loss_list, dim=0).transpose(0, 1).to(device)  # [B, K]
        norm_weights = losses_tensor    
        norm_weights = norm_weights.transpose(0, 1)               # [K, B]
        Vulnerability = [norm_weights[m] for m in range(K)]  

        W = [1 / K for _ in range(K)]

    for m, latent in enumerate(adv_out):
        delta = torch.randn_like(latent) * W[m]  * eps_list[m].view(B, 1, 1)
        delta_flat = delta.view(B, -1)
        delta_norm = torch.norm(delta_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
        eps_limit = eps_list[m] * W[m]  # [B, 1]
        scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(B, 1, 1)
        delta = delta * scale
        adv_out[m] = adv_out[m] + delta

    for _ in range(iters):

        for m, latent in enumerate(adv_out):
            adv_out[m] = adv_out[m].detach().clone().requires_grad_()

        out_model = model.forward_with_latentout(latentout=adv_out)
        loss = loss_function(out_model, j[-1].to(device).long())
        # import pdb; pdb.set_trace()
        model.zero_grad()
        loss.backward()

        for m in range(K):
            grad = adv_out[m].grad  # [B, N, D]
            grad_flat = grad.view(B, -1)
            norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True) + 1e-8  # [B, 1]

            adv_out_flat = latent_out[m].clone().reshape(adv_out[m].shape[0], -1)     # [B, T*D]
            adv_out_norm = torch.norm(adv_out_flat, p=2, dim=1)

            norm = norm.view(B, 1, 1)

            step_size = eps_list[m].view(B, 1, 1) * W[m] / iters
            step = step_size * grad / norm

            ep_grad.append((grad / norm).detach().clone().to('cpu'))

            delta = adv_out[m] + step - latent_out[m]
            delta_flat = delta.view(B, -1)
            delta_norm = torch.norm(delta_flat, p='fro', dim=1, keepdim=True)  # [B, 1]
            eps_limit = eps_list[m] * W[m]  # [B, 1]
            scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(B, 1, 1)
            delta = delta * scale
            adv_out[m] = latent_out[m] + delta
    # import pdb; pdb.set_trace()
    for m, latent in enumerate(adv_out):
        adv_out[m] = adv_out[m].detach()
    if test:
        torch.set_grad_enabled(False)
    if os.environ['method'] == "FGSM-MEP" and os.environ['Train'] == "True":
        return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)], ep_grad, W, [eps_list[m].detach().clone() for m in range(K)], Vulnerability
    if os.environ['method'] == 'FGSM-EP' and os.environ['Train'] == "True":
        return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)], W, Vulnerability
    if os.environ['method'] == 'FGSM-RS' and os.environ['Train'] == "True":
        return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)], Vulnerability

    return [(adv_out[m] - latent_out[m]).detach().clone() for m in range(K)]

def train(model,epochs,trains,valid,test,modalities,savedir,lr=0.001,weight_decay=0.0, optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss(),unsqueezing=[True,True], device="cuda:0",train_weights=[1.0,1.0],is_affect=[False,False],transpose=[False,False],ismmimdb=False,evalweights=None,recon=False, recon_weight=1, recon_criterion=torch.nn.MSELoss(),flips=-1, classesnum=[2,10,2,2],start_from=0,getattentionmap=False):

    optim = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    bestacc=0.0
    returnrecs=[]
    os.environ['Train'] = "False"
    
    ###
    perturb = {}
    mep_grad = {}
    ###
    import time
    
    for ep in range(epochs):
        
        toreturnrecs=[]
        totalloss=[]
        totals=[]
        fulltrains=[]
        indivcorrects=[]

        os.environ['Train'] = "True"
        start_time = time.time()
        for i in range(len(trains)):
            toreturnrecs.append([])
            count=0
            totalloss.append(0.0)
            totals.append(0)
            indivcorrects.append(0)    
            for j in trains[i]:
                #print('iter')
                if count >= len(fulltrains):
                    fulltrains.append({})
                if is_affect[i]:
                    jj=j[0]
                    if isinstance(criterion,torch.nn.CrossEntropyLoss):
                        jj.append((j[3].squeeze(1)>=0).long())
                    else:
                        jj.append(j[3])
                    fulltrains[count][str(i)]=jj

                else:
                    fulltrains[count][str(i)]=j
                if i == flips:
                    j[-1] = (j[-1] + 1) % classesnum[i]
                count += 1
        fulltrains.reverse()
        fulltrains=fulltrains[start_from:]

        # import pdb; pdb.set_trace()

        for js in tqdm(fulltrains):
            optim.zero_grad()
            losses=0.0
            for ii in js:
                #print(ii)
                model.to_logits=model.to_logitslist[int(ii)]
                indict={}
                for i in range(len(modalities[int(ii)])):
                    if unsqueezing[int(ii)]:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().unsqueeze(-1).to(device)
                    elif transpose[int(ii)]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[int(ii)][i]]=js[ii][i].float().to(device)
                for mod in indict:
                    indict[mod].requires_grad=True
                if recon:
                    # import pdb; pdb.set_trace()
                    out,rec = model(indict,use_recon=True)
                    stuffs = []
                    for modal in indict:
                        stuffs.append(torch.mean(indict[modal], dim=1))
                    origs=torch.cat(stuffs,dim=1)
                    loss=criterion(out,js[ii][-1].long().to(device))+ recon_weight*recon_criterion(rec,origs)
                else:
                    index = js[ii][-1].detach().clone()
                    js[ii] = js[ii][:-1]

                    if os.environ['Mode'] == "Attack" and ep>-1:
                        latent_out = model(indict, attack_mode = True)
                        # import pdb; pdb.set_trace()
                        if os.environ['method'] == "FGSM-RS":
                            adv_out, Vulnerability = FGSM(model,latent_out,j=js[ii], loss_function=criterion, device=device, iters=1)
                        elif os.environ['method'] == "FGSM-EP":
                            if ep == 0:
                                adv_out, W, Vulnerability = FGSM(model,latent_out,j=js[ii], loss_function=criterion, device=device, iters=1)
                            else:
                                # import pdb; pdb.set_trace()
                                init_delta = [[] for mod in range(len(modalities[int(ii)]))]
                                for ind in index:
                                    for mod in range(len(modalities[int(ii)])):
                                        init_delta[mod].append(perturb[ind.item()][mod])
                                # import pdb; pdb.set_trace()
                                init_delta = [torch.stack(mod) for mod in init_delta] 
                                adv_out, W, Vulnerability = FGSM_EP(model,latent_out,j=js[ii], loss_function=criterion, device=device, iters=1, init_delta=init_delta, Weight = W)
                        elif os.environ['method'] == "FGSM-PCO":
                            if ep == 0:
                                t_out, zero_init, adv_out, Vulnerability = FGSM_PCO(model,latent_out,j=js[ii], loss_function=criterion, device=device, iters=1)
                            else:
                                zero_init = None
                                init_delta = [[] for mod in range(len(modalities[int(ii)]))]
                                for ind in index:
                                    for mod in range(len(modalities[int(ii)])):
                                        init_delta[mod].append(perturb[ind.item()][mod])
                                # import pdb; pdb.set_trace()
                                init_delta = [torch.stack(mod) for mod in init_delta] 
                                t_out, adv_out, Vulnerability = FGSM_PCO(model,latent_out,j=js[ii], loss_function=criterion, device=device, iters=1, init_delta=init_delta)
                        elif os.environ['method'] == "FGSM-MEP":
                            if ep == 0:
                                adv_out, ep_grad, W, eps_list, Vulnerability = FGSM(model,latent_out,j=js[ii], loss_function=criterion, device=device, iters=1)
                            else:
                                init_delta = [[] for mod in range(len(modalities[int(ii)]))]
                                for ind in index:
                                    for mod in range(len(modalities[int(ii)])):
                                        init_delta[mod].append(perturb[ind.item()][mod])
                                init_delta = [torch.stack(mod) for mod in init_delta] 
                                adv_out, ep_grad, W, eps_list, Vulnerability = FGSM_EP(model,latent_out,j=js[ii], loss_function=criterion, device=device, iters=1, init_delta=init_delta, Weight=W)
                        if os.environ['method'] == "FGSM-MEP":
                            for m, ind in enumerate(index):
                                if ep == 0:
                                    if isinstance(W[0], torch.Tensor):
                                        W = [W[mod].item() for mod in range(len(adv_out))]
                                    mep_grad[ind.item()] = [ep_grad[mod][m].detach().clone().to('cpu') for mod in range(len(adv_out))]
                                    perturb[ind.item()] = [W[mod] * (eps_list[mod][m].to('cpu')*mep_grad[ind.item()][mod]/torch.norm(mep_grad[ind.item()][mod],p=2)) for mod in range(len(adv_out))]
                                    
                                    for mod in range(len(adv_out)):
                                        # import pdb; pdb.set_trace()
                                        delta = perturb[ind.item()][mod].to(device)
                                        # B = perturb[ind.item()][mod].shape[0]
                                        delta_flat = delta.view(-1)
                                        delta_norm = torch.norm(delta_flat, p='fro', dim=0, keepdim=True)  # [B, 1]
                                        eps_limit = eps_list[mod][m].to(device) * W[mod]  # [B, 1]
                                        scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(1, 1)
                                        delta = delta * scale
                                        perturb[ind.item()][mod] = delta.detach().clone().to('cpu')
                                else:
                                    if isinstance(W[0], torch.Tensor):
                                        W = [W[mod].item() for mod in range(len(adv_out))]
                                    # W = [1 / len(adv_out) for _ in range(len(adv_out))]
                                    mu = 0.3
                                    mep_grad[ind.item()] = [(mep_grad[ind.item()][mod] * mu + ep_grad[mod][m].detach().clone().to('cpu')) for mod in range(len(adv_out))]
                                    perturb[ind.item()] = [(perturb[ind.item()][mod] + W[mod] * eps_list[mod][m].to('cpu')*mep_grad[ind.item()][mod]/torch.norm(mep_grad[ind.item()][mod],p=2)) for mod in range(len(adv_out))]
                                    for mod in range(len(adv_out)):
                                        # import pdb; pdb.set_trace()
                                        delta = perturb[ind.item()][mod].to(device)
                                        # B = perturb[ind.item()][mod].shape[0]
                                        delta_flat = delta.view(-1)
                                        delta_norm = torch.norm(delta_flat, p='fro', dim=0, keepdim=True)  # [B, 1]
                                        eps_limit = eps_list[mod][m].to(device) * W[mod]  # [B, 1]
                                        scale = torch.clamp(eps_limit / (delta_norm + 1e-8), max=1.0).view(1, 1)
                                        delta = delta * scale
                                        perturb[ind.item()][mod] = delta.detach().clone().to('cpu')
                        else: 
                            for m, ind in enumerate(index):
                                perturb[ind.item()] = [adv_out[mod][m].detach().clone().to('cpu') for mod in range(len(adv_out))]
                        out = model.forward_with_latentout(latentout =  [(adv_out[iii]+latent_out[iii]) for iii in range(len(adv_out))]) 
                        probs = torch.softmax(out, dim=1).detach().clone()
                        # import pdb; pdb.set_trace()
                        lambda_ep = probs[torch.arange(out.size(0)), js[ii][-1].long()].view(-1, 1, 1)

                        if os.environ['method'] == "FGSM-PCO":
                            # import pdb; pdb.set_trace()
                            if zero_init == None:
                                # import pdb; pdb.set_trace()
                                train_out = [init_delta[mod].to(device) * lambda_ep + t_out[mod] * (1 - lambda_ep) for mod in range(len(adv_out))]
                            else:
                                train_out = [zero_init[mod] * lambda_ep + t_out[mod] * (1 - lambda_ep) for mod in range(len(adv_out))]
                            out = model.forward_with_latentout(latentout =  [(train_out[iii]+latent_out[iii]) for iii in range(len(adv_out))]) 
                        # import pdb; pdb.set_trace()
                    else:
                        out=model(indict)
                    if ismmimdb:
                        loss=criterion(out,js[ii][-1].float().to(device))
                    else:
                        loss=criterion(out,js[ii][-1].long().to(device))
                    if os.environ['AT_Methods'] == 'VARMAT' and (os.environ['method'] == "FGSM-RS" or os.environ['method'] == "FGSM-EP" or os.environ['method'] == "FGSM-MEP" or os.environ['method'] == "FGSM-PCO"):
                        Reg_item = torch.stack(Vulnerability, dim=1)           # [B, K]
                        
                        K = Reg_item.shape[1]
                        Avg_Weight = torch.full_like(Reg_item, 1.0 / K)

                        kl_loss = Reg_item.sum()/Reg_item.shape[0]

                        alpha = 1000
                        loss = alpha * kl_loss + loss

                    if os.environ['method'] == "FGSM-PCO":
                        train_prob = torch.softmax(out, dim=1).detach().clone()
                        if zero_init == None:
                            out = model.forward_with_latentout(latentout =  [(init_delta[iii].to(device)+latent_out[iii]) for iii in range(len(adv_out))]) 
                        else:
                            out = model.forward_with_latentout(latentout =  [(zero_init[iii]+latent_out[iii]) for iii in range(len(adv_out))]) 
                        last_probs = torch.softmax(out, dim=1).detach().clone()
                        beta = 10
                        reg_loss = ( beta * (((probs - last_probs) ** 2).mean() - ((train_prob - probs) ** 2).mean()) )
                        loss = loss +  reg_loss
                losses += loss*train_weights[int(ii)]
                total=len(js[ii][0])
                totals[int(ii)] += total
                totalloss[int(ii)] += loss.item()*total
                for i in range(total):
                    if torch.argmax(out[i]).item() == js[ii][-1][i]:
                        indivcorrects[int(ii)] += 1
            losses.backward()
            optim.step()
            #print("We're at "+str(totals))
        end_time = time.time()
        for ii in range(len(trains)):
            acc = float(indivcorrects[ii])/totals[ii]
            print("epoch "+str(ep)+" train loss dataset " +str(ii)+": "+str(totalloss[ii]/totals[ii]) + " acc: " +str(acc))
            toreturnrecs[ii].append(acc)
        os.environ['Train'] = "False"
        with torch.no_grad():
            accs=0.0
            for ii in range(len(valid)):
                totalloss=0.0
                totals=0
                corrects=0
                trues=[]
                preds=[]
                for jj in valid[ii]:
                    jj = jj[:-1]
                    j=jj
                    if is_affect[ii]:
                        j=jj[0]
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            j.append((jj[3].squeeze(1)>=0).long())
                        else:
                            j.append(jj[3])
                    #if ismmimdb:
                    #    j[0]=j[0].transpose(1,2)
                    model.to_logits=model.to_logitslist[ii]
                    indict={}
                    for i in range(len(modalities[ii])):
                        if unsqueezing[ii]:
                            indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                        elif transpose[ii]:
                            indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                        else:
                            indict[modalities[ii][i]]=j[i].float().to(device)
                    if os.environ['Mode'] == "Attack" and ep>-1:
                        latent_out = model(indict, attack_mode = True)
                        adv_out = FGSM(model,latent_out,j=j, loss_function=criterion, device=device, iters=eval(os.environ['adv_iter']))
                        model_out = model.forward_with_latentout(latentout =  [(adv_out[iii]+latent_out[iii]) for iii in range(len(adv_out))])
                        out=model(indict)
                    else:
                        out=model(indict)

                    if ismmimdb:
                        loss=criterion(out,j[-1].float().to(device))
                    else:
                        if os.environ['Mode'] == "Attack":
                            loss=criterion(model_out,j[-1].long().to(device))
                        else:
                            loss=criterion(out,j[-1].long().to(device))
                    totalloss += loss.item()*len(j[0])
                    if ismmimdb:
                        trues.append(j[-1])
                        preds.append(torch.sigmoid(out).round())
                        totals += len(j[-1])
                    else:
                        for i in range(len(out)):
                            if isinstance(criterion,torch.nn.CrossEntropyLoss):
                                preds=torch.argmax(out,dim=1)
                                if preds[i].item()==j[-1].long()[i].item():
                                    if os.environ['Mode'] == "Attack" and ep>-1:
                                        adv_preds=torch.argmax(model_out,dim=1)
                                        if adv_preds[i].item()==j[-1].long()[i].item():
                                            corrects += 1
                                    else:
                                        corrects += 1
                            else:
                                print(out[i].item(), j[-1][i].item())
                                if (out[i].item() >= 0) == j[-1].long()[i].item():
                                    if os.environ['Mode'] == "Attack" and ep>-1:
                                        if (adv_out[i].item() >= 0) == j[-1].long()[i].item():
                                            corrects += 1
                                    else:
                                        corrects += 1
                            totals += 1
                if ismmimdb:
                    true=torch.cat(trues,0)
                    pred=torch.cat(preds,0)
                    f1_micro = f1_score(true, pred, average="micro")
                    f1_macro = f1_score(true, pred, average="macro")
                    accs = f1_macro
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
                else:
                    acc=float(corrects)/totals
                    if evalweights is None:
                        accs += acc
                    else:
                        accs += acc*evalweights[ii]
                    print("epoch "+str(ep)+" valid loss dataset"+str(ii)+": "+str(totalloss/totals)+" acc: "+str(acc))
                    toreturnrecs[ii].append(acc)
            if accs > bestacc:
                print("save best")
                bestacc=accs
                torch.save(model,savedir)
            elif ep == epochs-1:
                lastdir = savedir[:-3] + "_last.pt"
                torch.save(model,lastdir)
            returnrecs.append(toreturnrecs)
    model=torch.load(savedir, map_location='cpu').to(device)
    testaccs=[]
    with torch.no_grad():
        rets=[[],[],[],[]]
        for ii in range(len(test)):
            model.to_logits=model.to_logitslist[ii]
            totals=0
            corrects=0
            trues=[]
            preds=[]
            for jj in test[ii]:      
                jj = jj[:-1]      
                j=jj
                if is_affect[ii]:
                    j=jj[0]
                    j.append((jj[3].squeeze(1) >= 0).long())

                indict={}
                for i in range(0,len(modalities[ii])):
                    if unsqueezing[ii]:
                        indict[modalities[ii][i]]=j[i].float().unsqueeze(-1).to(device)
                    elif transpose[ii]:
                        indict[modalities[ii][i]]=j[i].float().to(device).transpose(1,2)
                    else:
                        indict[modalities[ii][i]]=j[i].float().to(device)
                if os.environ['Mode'] == "Attack":
                    latent_out = model(indict, attack_mode = True)
                    adv_out = FGSM(model,latent_out,j=j, loss_function=criterion, device=device, iters=eval(os.environ['adv_iter']))
                    model_out = model.forward_with_latentout(latentout =  [(adv_out[iii]+latent_out[iii]) for iii in range(len(adv_out))])
                    out=model(indict)
                else:
                    out=model(indict)

                if getattentionmap:
                    rets[ii].append(model.attns)
                    #break                       
                if ismmimdb:
                    trues.append(j[-1])
                    preds.append(torch.sigmoid(out).round())
                else:
                    for i in range(len(out)):
                        if isinstance(criterion,torch.nn.CrossEntropyLoss):
                            preds=torch.argmax(out,dim=1)
                            if preds[i].item()==j[-1].long()[i].item():
                                if os.environ['Mode'] == "Attack":
                                    adv_preds=torch.argmax(model_out,dim=1)
                                    if adv_preds[i].item()==j[-1].long()[i].item():
                                        corrects += 1
                                else:
                                    corrects += 1
                        else:
                            print(out[i].item(), j[-1][i].item())
                            if (out[i].item() >= 0) == j[-1].long()[i].item():
                                if os.environ['Mode'] == "Attack":
                                    if (adv_out[i].item() >= 0) == j[-1].long()[i].item():
                                        corrects += 1
                                else:
                                    corrects += 1
                        totals += 1

            if ismmimdb:
                true=torch.cat(trues,0)
                pred=torch.cat(preds,0)
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                #accs = f1_macro
                print("test f1_macro: "+str(f1_macro)+" f1_micro: "+str(f1_micro))
            elif not getattentionmap:
                acc=float(corrects)/totals
                testaccs.append(acc)
                print("test acc dataset "+str(ii)+": "+str(ii)+" "+str(acc))
    if getattentionmap:
        return rets
    return testaccs



