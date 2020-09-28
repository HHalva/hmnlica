# Hidden Markov Nonlinear ICA: Unsupervised Learning from Nonstationary Time Series.

This repository contains code for our [Hidden Markov Nonlinear ICA model](https://arxiv.org/abs/2006.12107), published at UAI 2020. Work done by Hermanni Hälvä (University of Helsinki) and Aapo Hyvärinen (University of Helsinki). The algorithm here is the stochastic subchain sampling approach discussed in Section 3.3. of our paper, which corresponds to minibatch training of the HMM.

## Dependencies
We have tested the code on:
- Python 3.7.5
- JAX 0.1.55
- jaxlib 0.1.37
- numpy 1.17.4
- scipy 1.3.2
- scikit-learn 0.21.3
- pickle 4.0

## References

If you use our code/model for your research, we would be grateful if you cite our work as:

```bib
@InProceedings{pmlr-v124-halva20a,
  title = 	 {Hidden Markov Nonlinear ICA: Unsupervised Learning from  Nonstationary Time Series},
  author =       {H\"{a}lv\"{a}, Hermanni and Hyv\"{a}rinen, Aapo},
  pages = 	 {939--948},
  year = 	 {2020},
  volume = 	 {124},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Virtual},
  month = 	 {03--06 Aug},
  publisher =    {PMLR},
}
```

## License
A full copy of the license can be found [here](LICENSE)
