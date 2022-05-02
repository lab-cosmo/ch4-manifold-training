Generates "quasi-constant" manifolds following the procedure from DOI:10.1063/5.0070488, 
showing the relationship to the discrete degeneracies from DOI:10.1103/PhysRevLett.125.166001
and the degenerate singularities of DOI:10.12688/openreseurope.14156.1 .

Uses a simple multi-layer perceptron to demonstrate _interpolative_ learning on the manifold.

Build docker image:
docker build --tag reply-reproduce .

Run docker:
docker run -p 8888:8888 reply-reproduce

Next go to localhost:8888 in the browser and open terminal.

To generate a set of quasi-constant manifolds based on librascal-computed SOAP features, simply run 

$ python3 degenerate_orbits.py

This will generate a `quasiconstant-manifolds.pickle` containing the structures for different SOAP parameters.

Open terminal in jupyter notebook and run the following commands to fit NNs and save predictions:

$ python3 nn_fitting.py 201 10 80 using_full-x10_manifold.npz data
$ python3 nn_fitting.py 20 1 80 using_20_from_manifold.npz data
$ python3 nn_fitting.py 0 0 80 not_using_manifold.npz data

Next open plotting.ipynb to do plots.
