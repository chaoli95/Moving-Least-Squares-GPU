# Moving-Least-Squares-GPU

## Compile

GPU version: 

```
nvcc -o mls mls.cu -lcusolver -lcusparse
```
## Usage

```
./mls data/sphere.off
```

## Changing parameters

### Changing sampling resolutions: 

change NX, NY, NZ on line 19-21

### Changing weight function radius

change H on line 33

## TODO

1. C++11 style IO
2. Add algorithm description in README
3. Design the algorithm when space needed is larger than GPU memory (in other words, batching)
