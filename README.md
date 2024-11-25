# ICAN

Ican is fast, basic neural network library. It's designed for experimental usages, not recommended to use commercial.

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. See deployment
for notes on deploying the project on a live system.

### Prerequisites

Requirements for the software and other tools to build, test and push

- [libpng](http://www.libpng.org/pub/png/libpng.html)
- [libjpeg](http://www.ijg.org/)
- [cjson](https://github.com/DaveGamble/cJSON)

### Installing

A step by step series of examples that tell you how to get a development
environment running

Clone project

    git clone https://github.com/omerfdmrl/ican ican

Run example network

    make run SOURCE=./examples/and.c

End with an example of training of AND gate.

### Usage

You can check `examples/` folder.

About makefile;
  - `make all`: create library
  - `make run`: run custom main.c
  - `make clean`: delete library file
  - `make fclean`: delete library file and object files
  - `make gtest`: run gdb test
  - `make utest`: run custom test from test/ folder
  - `make memcheck`: run valgrind for detect memory leaks

About folder hierarchy;
 - `test/`: custom test functions and unity library
 - `src/`: library functions and header file
 - `examples/`: examples usages of library
 - `build/depends/`: dependencies of library
 - `build/objs/`: stores object files while creating library (`make all`)
 - `build/results/`: includes bin file (gdb test), output file (program) and library file 

### Roadmap

- [x] Initializers
  - [x] Ones
  - [x] Random HeUniform
  - [x] Random Normal
  - [x] Random Uniform
  - [x] Random Xavier
  - [x] Zeros
- [x] Activation
  - [x] Sigmoid
  - [x] Tanh
  - [x] Softmax
- [ ] Layers
  - [x] Activation
  - [x] Dense
  - [x] Dropout
  - [x] Shuffle
  - [x] Max Pooling
  - [x] Min Pooling
  - [x] Mean Pooling
  - [ ] Flatten
  - [x] RNN
  - [x] GRU
- [ ] Models
  - [x] Sequential
  - [ ] GAN
- [ ] Optimizers
  - [x] Finite Diff
  - [ ] Gradient Descent
    - [x] Batch Gradient Descent
    - [ ] Stochastic Gradient Descent
    - [ ] Mini-Batch Gradient Descent 
- [ ] Transformers
    - [x] Encoder
    - [ ] Decoder
    - [ ] Model support
        - [ ] GPT2
        - [ ] Llama  
- [x] Utils
  - [x] CSV
  - [x] IMG
  - [x] IO
- [ ] TODO
  - [ ] Function for split test-train data
  - [ ] 3D Layers
  - [ ] 3D Models

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/omerfdmrl/ican/tags).

## Authors

- **Ömer Faruk Demirel** - _Main Developer_ -
  [omerfdmrl](https://github.com/omerfdmrl)
- **Billie Thompson** - _Provided README Template_ -
  [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/omerfdmrl/ican/contributors)
who participated in this project.

## License

This project is licensed under the [MIT](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details
