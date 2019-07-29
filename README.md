Compressed sensing (also known as compressive sensing, compressive sampling, or sparse sampling)
is a signal processing technique for efficiently acquiring and reconstructing a signal, by finding
solutions to underdetermined linear systems. This is based on the principle that, through optimization,
the sparsity of a signal can be exploited to recover it from far fewer samples than required by the
Nyquist–Shannon sampling theorem. There are two conditions under which recovery is possible. The first
one is sparsity, which requires the signal to be sparse in some domain. The second one is incoherence
, which is applied through the isometric property, which is sufficient for sparse signals.

In this tutorial I’ll be investigating compressed sensing in Python. Since the idea of compressed sensing
can be applied in wide array of subjects, I’ll be focusing mainly on how to apply it in one and two dimensions
to things like sounds and images (3-D compressive sampling can easily be implemented by using the same approaches).
Specifically, I will show how to take a highly incomplete data set of signal samples and reconstruct the
underlying sound or image. It is a very powerful technique.
