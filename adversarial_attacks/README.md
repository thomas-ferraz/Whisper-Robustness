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

<img src="adversarial_attacks/data/wer_model_snr.pdf"
     alt="Results"
     style="float: left; margin-right: 10px;" />

## Samples
Here you can hear some samples for some of the models.
