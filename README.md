#  Halcyon

![halcyon](https://user-images.githubusercontent.com/6816040/100544654-9c166d80-329a-11eb-86e9-e1dd17a496a6.png)


[Halcyon: an accurate basecaller exploiting an encoder-decoder model with monotonic attention.](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btaa953/5962086?searchresult=1)

Halcyon incorporates neural-network techniques frequently used in the field
of machine translation, and employs monotonic-attention mechanisms to learn semantic
correspondences between nucleotides and signal levels without any pre-segmentation against input
signals.

## Installation


### pip

```bash
pip install pyhalcyon
```
### bioconda

```bash
conda install pyhalcyon
```

### Docker

```bash
docker pull relastle/halcyon
```

## Basic Usage

Halcyon basecalls from fast5 files in a single directory and outputs the basecalled sequences into a single fasta file.

Suppose that you have the following directory structure.

```
./test_data/
└── signals.fast5
```

The commands below are for obtaining fasta file (`./test.fasta`) as following.

```
>./test_data/signals.fast5
TGG...TAT
```

### pip /bioconda / from source

```bash
halcyon basecall -i ./test_data/ -o ./test.fasta
```

### docker

```bash
docker run -it --rm -v <from>:<to> relastle/halcyon basecall -i ./test_data/ -o ./test.fasta
```

⚠️  Please note that you should mount the host directory into the container appropriately.


## Road Map

- [x] Inference module.
- [ ] From-scratch training module.
- [ ] Retraining module.
- [ ] Visualizing a attention transition.

## Citation

Below is bibtex format for citation.

```bibtex
@article{10.1093/bioinformatics/btaa953,
    author = {Konishi, Hiroki and Yamaguchi, Rui and Yamaguchi, Kiyoshi and Furukawa, Youichi and Imoto, Seiya},
    title = "{Halcyon: An Accurate Basecaller Exploiting An Encoder-Decoder Model With Monotonic Attention}",
    journal = {Bioinformatics},
    year = {2020},
    month = {11},
    abstract = "{In recent years, nanopore sequencing technology has enabled inexpensive long-read sequencing, which promises reads longer than a few thousand bases. Such long-read sequences contribute to the precise detection of structural variations and accurate haplotype phasing. However, deciphering precise DNA sequences from noisy and complicated nanopore raw signals remains a crucial demand for downstream analyses based on higher-quality nanopore sequencing, although various basecallers have been introduced to date.To address this need, we developed a novel basecaller, Halcyon, that incorporates neural-network techniques frequently used in the field of machine translation. Our model employs monotonic-attention mechanisms to learn semantic correspondences between nucleotides and signal levels without any pre-segmentation against input signals. We evaluated performance with a human whole-genome sequencing dataset and demonstrated that Halcyon outperformed existing third-party basecallers and achieved competitive performance against the latest Oxford Nanopore Technologies’ basecallers.The source code (halcyon) can be found at https://github.com/relastle/halcyon.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa953},
    url = {https://doi.org/10.1093/bioinformatics/btaa953},
    note = {btaa953},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaa953/34178609/btaa953.pdf},
}
```
