# Robustness to Adversarial Attacks

We used FGSM to create adversarial samples of the 
[FLEURS](https://huggingface.co/datasets/google/fleurs) dataset. Since FGSM 
modification for an input depends on its gradients, to have standard degradations
we fixed the SNR for each sample according to the equation below.

$$
SNR(\epsilon,x)=20\cdot\log\Big(\frac{||x||_2}{||\delta(\epsilon)||_p}\Big)
$$

We tested three different samples no degradation, SNR=40, SNR=35. We evaluated 
the impact on the robustnes of the size of the model and the availability of the 
language. 

## Results
Global results for the experiments can be seen in the figure below. 

![image](https://github.com/thomas-ferraz/Whisper-Robustness/assets/8398479/dbaaee00-7e89-4000-a330-a9f42a72eba3)

## Samples
Here you can hear some result samples.

### Tiny
||Clean|SNR=40|SNR=35|
|---|---|---|---|
|FR|[FR-Tiny-Clean.wav](adversarial_attacks/data/samples/FR-Tiny-Clean.wav)|[FR-Tiny-40.wav](adversarial_attacks/data/samples/FR-Tiny-40.wav)|[FR-Tiny-35.wav](adversarial_attacks/data/samples/FR-Tiny-35.wav)|
|`de nombreux formats courants famille de formats aps par exemple sont egaux a ce rapport d aspect ou s en approchent de pres`|`de nombreux formes d accordant famille de formes a ps par exemple sans egot a sur rapport d aspect au sein rapproche de vrai`|`nous sommes au center de la france a l ontario a l ontario`|`nous sommes au mata rain ce que nous sommes a la fin de la fin de la fin de la fin de la fin de la fin de la fin de la fin de la fin de la fin de la fin de la fin de la fin de la fin de...`|




