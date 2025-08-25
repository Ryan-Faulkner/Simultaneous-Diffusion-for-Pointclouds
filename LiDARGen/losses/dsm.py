import torch

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)

def anneal_dsm_score_estimation_simultaneous(scorenet, perturbed_samples, used_sigmas, noise, masks, labels=None, anneal_power=2., hook=None):
    #labels is the number of times noise is added
    #if none provided, pickls a random number between 0 and max number of sigmas, for each sample in batch (stamples.shape[0]).
    #I need to change this, easiest method is.... for loop outside that calls this function passing in labels from len(sigmas) backwards to 0
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples.float(), labels)
    masks = masks.view(masks.shape[0], -1)
    masks = torch.tile(masks,(1,2))
    target = target.view(target.shape[0], -1)
    scoresForLoss = scores.view(scores.shape[0], -1)
    # sky = sky.view(sky.shape[0], -1)[masks]
    numPixels = masks.sum()
    # numSky = sky.sum()
    #Multiply by masks to remove pixels which are unavailable
    #multiply by masks.shape[-1] then divide by num Pixels such that:
    # If all pixels are available, no difference
    # If half pixels are available, multiplied by 2, to account for the ,sum() being half as large.
    #Ok so if I remove the second part, I no longer weight the loss based on how much noise is being applied
    loss = 1 / 2. * (((masks * (scoresForLoss - target)) ** 2).sum(dim=-1)*masks.shape[-1] / numPixels) * used_sigmas.squeeze() ** anneal_power
    # loss = 1 / 2. * (((masks * (scoresForLoss - target)) ** 2).sum(dim=-1)*masks.shape[-1] / numPixels) * used_sigmas.squeeze() ** anneal_power

    # print(numPixels)
    # print(scores.shape)
    # print(target.shape)
    # print(masks.shape)
    # diff =(masks * (scores - target)).sum(dim=-1)
    # squared = ((scores - target) ** 2).sum(dim=-1)
    # then = diff * used_sigmas.squeeze()
    # plus = then ** anneal_power
    # resultingIn = 1 / 2. * plus
    # print(diff)
    # print(squared)
    # print(then)
    # print(plus)
    # print(resultingIn)
    # print(loss) 
    # print(loss.shape)
    # loss2 = 1 / 2. * ((scores[sky] - target[sky]) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    # loss = (loss+loss2*numSky)/numPixels
    # loss = loss

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0), scores


def anneal_dsm_score_estimation_with_mask(scorenet, perturbed_samples, used_sigmas, noise, masks, sky, sigmas, labels=None, anneal_power=2., hook=None):
    #labels is the number of times noise is added
    #if none provided, pickls a random number between 0 and max number of sigmas, for each sample in batch (stamples.shape[0]).
    #I need to change this, easiest method is.... for loop outside that calls this function passing in labels from len(sigmas) backwards to 0
    if labels is None:
        labels = torch.randint(0, len(sigmas), (perturbed_samples.shape[0],), device=perturbed_samples.device)
    if(used_sigmas is None):
        used_sigmas = sigmas[labels].view(perturbed_samples.shape[0], *([1] * len(perturbed_samples.shape[1:])))
        noise = torch.randn_like(perturbed_samples) * used_sigmas
        perturbed_samples = perturbed_samples + noise
    #This line takes the corresponding noise from the precalculated noise at each timestep
    #I...need to merge this with an also-passed-in prior prediction
    # used_sigmas = sigmas[labels].view(perturbed_samples.shape[0], *([1] * len(perturbed_samples.shape[1:])))
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples.float(), labels)
    masks = masks.view(masks.shape[0], -1)
    target = target.view(target.shape[0], -1)
    scoresForLoss = scores.view(scores.shape[0], -1)
    # sky = sky.view(sky.shape[0], -1)[masks]
    numPixels = masks.sum()
    # numSky = sky.sum()
    #Multiply by masks to remove pixels which are unavailable
    #multiply by masks.shape[-1] then divide by num Pixels such that:
    # If all pixels are available, no difference
    # If half pixels are available, multiplied by 2, to account for the ,sum() being half as large.
    #Ok so if I remove the second part, I no longer weight the loss based on how much noise is being applied
    loss = 1 / 2. * (((masks * (scoresForLoss - target)) ** 2).sum(dim=-1)*masks.shape[-1] / numPixels) * used_sigmas.squeeze() ** anneal_power
    # loss = 1 / 2. * (((masks * (scoresForLoss - target)) ** 2).sum(dim=-1)*masks.shape[-1] / numPixels) * used_sigmas.squeeze() ** anneal_power

    # print(numPixels)
    # print(scores.shape)
    # print(target.shape)
    # print(masks.shape)
    # diff =(masks * (scores - target)).sum(dim=-1)
    # squared = ((scores - target) ** 2).sum(dim=-1)
    # then = diff * used_sigmas.squeeze()
    # plus = then ** anneal_power
    # resultingIn = 1 / 2. * plus
    # print(diff)
    # print(squared)
    # print(then)
    # print(plus)
    # print(resultingIn)
    # print(loss) 
    # print(loss.shape)
    # loss2 = 1 / 2. * ((scores[sky] - target[sky]) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    # loss = (loss+loss2*numSky)/numPixels
    # loss = loss

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0), scores

#Pass in the Sky mask as well, and use it to massively reduce loss caused by sky pixels being way off.
