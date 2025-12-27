# Graceful Prüfer Codes

This repository contains source code and datasets related to the study of graceful labelings of trees represented through Prüfer codes.

A tree on n vertices admits a graceful labeling if its vertices can be labeled by distinct integers such that the absolute differences on edges form exactly the set {1, 2, ..., n−1}. In many classical works, graceful labelings are considered primarily as a property of a tree, and attention is often focused on existence or enumeration results. In this project, the emphasis is placed instead on individual graceful labelings and their explicit representations.

The approach taken here is based on the use of Sheppard codes. Each Sheppard code determines a labeled graph via a fixed edge construction. When the resulting graph is a tree, the labeling obtained in this way is graceful by construction. Such trees are then encoded by their Prüfer codes. This provides a direct representation of graceful labelings as elements of Prüfer space.

The datasets provided in this repository consist of Prüfer codes obtained through this procedure. Vertex labels are taken from the set {0, 1, ..., n−1}, and each line of a dataset corresponds to a single Prüfer code of length n−2. In this representation, each Prüfer code corresponds to a specific graceful labeling, rather than merely to an unlabeled tree.

The generation procedure is algorithmic and reproducible. For a fixed number of vertices n, a prescribed subset of Sheppard codes is enumerated, converted into graphs, filtered to retain trees, and then transformed into Prüfer codes. The total numbers obtained in this way agree with known results on the number of graceful trees.

The purpose of providing explicit Prüfer codes is to make individual graceful labelings available in a concrete and standard representation. This allows graceful labelings to be handled in the same way as general labeled trees in Prüfer space, using familiar combinatorial and computational tools.

This repository is part of a bachelor's thesis and is released to support reproducibility and further examination of graceful labelings in Prüfer representation.

## License

This project is released under the MIT License.
