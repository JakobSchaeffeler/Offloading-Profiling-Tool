This tool consists of three executables: 

## `run_benchmarks.py`
Pofiles the executable passed as an argument and puts the output into files. Additionally the programming model, GPU vendor and subdirectory where the results should be stored have to specified    

### Example
```
python3 run_benchmarks.py --gpu nvidia --model cuda -o measurements_cuda_triad <cuda_triad_executable>
```

## `create_profiles.py`
Processes the collected metrics and stores them into files for each kernel seperatly.

### Example
```
python3 create_profiles.py --gpu nvidia --model cuda -o results_cuda_triad <directory with profiles creater by runbenchmarks.py>
```

## `signature.py`
Pass an arbitrary amount of files created by *create_profiles.py*. Creates signature plots from that.

### Example usage

```
python signature.py --name <name of plot> <base-kernel-file> <arbitrarily many kernel files>
```

This will create a plot with the specified name with all outputs realative to the first passed signature file. Additionally thread/team counts will be printed to the terminal.
