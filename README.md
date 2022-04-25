# Deep Clustering for Unsupervised Learning of Visual Features

Pre-trained convolutional neural nets, or covnets produce excelent general-purpose features that can be used to improve the generalization of models learned on a limited amount of data. Moving forward, we are always in search of bigger, better and more diverse datasets, which ould require a tremendous amount of manual annotations, despite the expert knowledge in crowdsourcing  ccumulated by the community over the
years. So, this project attempts to review methods which can be trained on internet-scale datasets, with potentially billions of images with no supervision. 

The implenentation here is largely inspired from [Facebook AI Research paper: Deep Clustering for Unsupervised Learning of Visual Features](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf).

## Literature Review
### Unsupervised Learning of Features
Several methods exist that use unsupervised learning to pre-train covnets, for example:
* Coates and Ng method, uses k-means and learns each layer sequentially going bottom-up. 
* Yang et al., whcih iteratively learns covnet fratures and clustes with a recurrent framework. Although this works wonders on small datasets, it is not scalable to lager ones.
* Bojanowski and Joulin learn visual features on a large dataset, with a loss that attempts to preserve the information flowing through the network. 
* And finally, Caron et al., the approach we follow, uses k-means in an end-to-end fashion.

### Self-supervised Learning
We can use pretext tasks to replace the labels annotated by humans by “pseudo-labels” directly computed from the raw input data.

### Generative Models
Recently, unsupervised learning has been making a lot of progress on image generation. Typically, a parametrized mapping is learned between a predefined random noise and the images, with either an autoencoder, a generative adversarial network (GAN) or more directly with a reconstruction loss.

## Our Approach
### Preliminaries
### Unsupervised learning by clustering
### Trivial Solutions and how to avoid them

## Implementation Details

## Visualisations

## Conclusion and Further Study

## References
* [Facebook AI Research: Caron, Bajanowski, Joulin and Douze; Deep Clustering for Unsupervised Learning of Visual Features](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf)
* Bottou, L.: Stochastic gradient descent tricks. In: Neural networks: Tricks of the trade, pp. 421–436. Springer (2012)
* Huang, F.J., Boureau, Y.L., LeCun, Y., et al.: Unsupervised learning of invariant feature hierarchies with applications to object recognition. In: CVPR (2007)
* [ Image Colorization With Deep Learning and Classification](https://cs.stanford.edu/people/karpathy/deepvideo/)
* Bojanowski, P., Joulin, A.: Unsupervised learning by predicting noise. ICML (2017)
* Yang, J., Parikh, D., Batra, D.: Joint unsupervised learning of deep representations
and image clusters. In: CVPR (2016)