Questions
- Just wondering because I have no idea, is normalising linearly best or would something like a logarithmic scale be better for the radiologists? Same for X-Rays
- In particular with the X-Ray VQVAE, is it actually a good idea to use a discriminator? On one hand realism isn't as important here so better mode coverage is arguably better. But on the other hand, adding a discriminator has been shown to improve reconstruction quality because the codes are able to store much more information. Perhaps both could be tried and compared purely on L1/L2 loss and see which one does better

TODO
- [ ] Think about how best to store CT scans. 273mb * batch size 16 = 4.4gb. So looking at least 4gb/s read speeds, realistically more like >8gb/s, which is insane. Can zip compress as .npz which drops it by 4x. Could maybe preprocess to max side length 256 which might get another factor of 8.
    - [x] For now I have interpolated them all down by a factor of 2 i.e. '''F.interpolate(x_torch.unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode="trilinear")''' and zip compressed as .npz which has reduced the sizes down by a factor ~20x.
    - [ ] Save as video files and load with NVIDIA Dali?
            Currently NVIDIA Dali process 3D data by using gpu direct storage. I tried loading and perfoming preprocessing steps using CuPy as we can apply the preprocessing steps in gpu. This seems to achieve an increase of ~5% gpu usage increase. Might be worth using a cache subset of the data as MONAI does.
- [ ] Play around with the sizes of CT latents. Currently 8x16x16=2048 which is quite large. But 8x8x8=512 is quite small. Maybe 16x8x8 could be better?
- [ ] For 3D VQGAN maybe try the architecture used for video diffusion models with 2D convolutions and 1D (flash) attention down the other dimension?
- [ ] Inspired by the video diffusion models above, could break convolutions into three 2D convolutions per block. Probably 3 sets. For the 3 spatial dims (1,2,3) can move one dim into batch and do convolutions on (1,2), (1,3), (2,3). 
- [ ] Add dropout to help generalisation
- [ ] Add option for AdamW
- [x] Try augmenting images before passing to autoencoder. Might help with generalisation. 
        Used [kornia](https://github.com/kornia/kornia) as it has differentiable transformations. I added geometric X-ray and CT augmentations. This helped a lot in the knee dataset as training was failing in the X-ray case. 
- [ ] If VQGAN trained with augmentations as above then try training latent model with distribution augmentation (i.e. condition latent model on the augmentation type/weight).
- [ ] Change all rearranging operations to einops
- [ ] Use flash attention for VQGAN attention. Might be especially helpful for 3D VQGAN
- [ ] Clean up VQGANs. Use the same resblock etcs for both but with 2d/3d convolutions 
- [ ] Implement max probability sampling
- [ ] Implement top-k sampling
- [ ] Implement MaskGIT sampling
- [ ] Try hourglass transformer
- [ ] Try ViT VQGAN (from Improved VQGANs)
- [ ] Should we really be using batchnorm in the discriminator? Probably especially not since there are two different kinds of x-rays
- [ ] Possible it might be helpful to train the X-ray VQGAN with all X-rays to improve generalisation even if not all of them are used at the latent level
- [ ] Condition VQGAN on view angle?
- [ ] Sort out logging
- [ ] Maybe try some other Vector-Quantization alternatives. See what the latest approaches are using
- [ ] Would it be good to pass all X-Rays into a single 3D VQGAN? (instead of separately to a 2D one)
- [ ] Change back to saving models every step if helpful? I need to clean up my storage so checkpoints overwrite each other for now.
- [ ] When using autoregressive sampler there's no reason for causal masking over context so fix that.
- [ ] Allow x-ray and ct encodings to have different emb_dim and still work with the latent transformer (by passing in separately and using different linear layers).
- [ ] Try conditioning transformer on time again now everything else is optimised.
- [ ] Train VQGANs with L2 loss instead of L1 to penalise deviation from the data more? (would need rescaling since squaring would make values smaller)
- [ ] Change the latent saving function to save x-ray latents as long values instead of full flaot vectors to save storage space. Relatively minor since the datasets are small though.
- [ ] Sample with fewer diffusion steps. The strong conditioning signal should mean we can skip quite a lot of steps at once.
