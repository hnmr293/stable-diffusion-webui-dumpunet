An extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that adds a custom script which let you to observe U-net feature maps.

# Example

Model Output Image:

![Model Output Image](images/00.png)

```
Model: waifu-diffusion-v1-3-float16 (84692140)
Prompt: a cute girl, pink hair
Sampling Method: DPM++ 2M Karras
Size: 512x512
CFG Scale: 7
Seed: 1719471015
```

U-net features:

- IN00 (64x64, 320ch)

Let the feature value is `v`, larger `|v|` is white, and zero is black.

step 1

![IN00 step1](images/IN00-step01.png)

step 10

![IN00 step10](images/IN00-step10.png)

step 20

![IN00 step20](images/IN00-step20.png)

- OUT02 (16x16, 1280ch)

step 20

![OUT02 step20](images/OUT02-step20.png)

- OUT11 (64x64, 320ch)

step 1

![OUT11 step1](images/OUT11-step01.png)

step 10

![OUT11 step10](images/OUT11-step10.png)

step 20

![OUT11 step20](images/OUT11-step20.png)

Color map mode:

Red means the value is positive, and blue means the value is negative.

![OUT11 step20 cm](images/OUT11-step20-cm.png)
