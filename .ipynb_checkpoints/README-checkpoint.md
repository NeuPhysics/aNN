ANN - Artificial Neural Network Method
=========================================================

Artificial neural network for differential equation solving.



Cotents
----------------------

I gave a talk about this idea a while ago. [A Physicist's Crash Course on Artificial Neural Network](https://github.com/NeuPhysics/sync-de-solver/blob/master/ipynb/neural-net.ipynb)

We did not use the simple back-prop method for the project because it's aweful. (I do not have the original code for it but I rewrote [an example using PyTorch](https://github.com/emptymalei/pytorch-differential-equation-solver/blob/master/graph-constructor.ipynb).) We really need much better cost minimization method. So we tested the best minimization algorithms here.

Structure of this repository:

```
.
├── LICENSE
├── MMA
│   ├── homogeneousGas.nb
│   └── vac.nb
├── README.md
├── ipynb
│   ├── Basics.ipynb
│   ├── Basics.ipynb.bak
│   ├── HomogeneousModel.ipynb
│   ├── NetworkConstructor.ipynb
│   ├── Untitled.ipynb
│   ├── Untitled1.ipynb
│   ├── ann_julia.ipynb
│   ├── assets
│   ├── test.ipynb
│   ├── vacOsc4Comp.ipynb
│   ├── vacOsc4CompSSConvention.ipynb
│   ├── vacOsc4Fourier.ipynb
│   ├── vacOsc4Piecewise.ipynb
│   ├── vacuum-Copy1.ipynb
│   ├── vacuum-Copy2.ipynb
│   ├── vacuum.ipynb
│   ├── vacuum4Component.ipynb
│   └── vacuumClean.ipynb
├── notes
│   └── note-2015S.pdf
└── py
    ├── functionvalue-moretol.txt
    ├── functionvalue.txt
    ├── ss
    ├── timespent-moretol.txt
    ├── timespent.txt
    ├── vacOsc4CompSSConvention-moretol.py
    ├── vacOsc4CompSSConvention-verify.py
    ├── xresult-1.txt
    ├── xresult-moretol.txt
    └── xresult.txt
```


0. `notes` is the notes for the project. I explained some of the conventions and the preliminary results. I pulled this file from my private repo of the project. I think it can made public now.
1. The folder `MMA` is for my Mathematica code related to this problem.
2. `ipynb` contains the Jupyter Notebooks.
   1. `Basics.ipynb`: the basics of the idea. quite similar to the talk mentioned above.
   2. `HomogeneousModel.ipynb`: solving Homogeneous gas model of neutrino oscillations.
   3. `NetworkConstructor.ipynb`: example of network constructor for differential equations.
   4. `ann_julia.ipynb`: Julia code example.
   5. `test.ipynb`: testing different methods, benchmarking functions.
   6. `vacOsc4Comp.ipynb`: Solving neutrino vacuum oscillations.
   7. `vacOsc4CompSSConvention.ipynb`: vacuum oscillations using Shanshak's convention
   8. `vacOsc4Fourier.ipynb`: Using Fourier as the internal network structure, aka, Fourier analysis as approximators.
   9. `vacOsc4Piecewise.ipynb`: Using piecewise functions as approximators
   10. `vacuumClean.ipynb`: Vacuum oscillations cleaned up
   11. `vacuum4Component.ipynb`: Vacuum oscillations with 4-component conventions
3. `py` folder is for the python code.


