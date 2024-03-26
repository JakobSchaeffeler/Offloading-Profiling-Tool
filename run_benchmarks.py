import sys
import os
import subprocess
from shutil import which
import argparse
import csv

is_nvidia = True

def main():

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Profiling tool for GPUs')

    # Add optional argument --gpu with choices of nvidia or amd
    parser.add_argument('--gpu', choices=['nvidia', 'amd'], help='Optional argument specifying GPU type (nvidia or amd)')

    # Add optional argument -o followed by a filename
    parser.add_argument('-o', metavar='OUTPUT_FILENAME', nargs=1, help='Optional subdir argument to store traces, default = measurements')

    # Optional Argument to choose offloading programming model, default omp
    parser.add_argument('--model', choices=['omp', 'hip', 'cuda'], default='omp', help='Optional argument specifying model (omp, hip, or cuda)')


    # Add non-optional argument for input file
    parser.add_argument('input', metavar='INPUT_FILENAME', help='Non-optional argument specifying input filename')

    # Parse the arguments
    args = parser.parse_args()


    # Try to detect if Nvidia GPU by testing if nvidia-smi is available, if not: amd
    global is_nvidia
    if args.gpu:
        if args.gpu[0] == "amd":
            is_nvidia = False
    else:
        if which("nvidia-smi") is None:
            is_nvidia = False

    subdir = "measurements"


    if args.o:
         subdir += "_" + args.o[0]

    if not os.path.exists(subdir):
        os.makedirs(subdir)
    if is_nvidia:
        #if which("ncu") is None:
        #    raise FileNotFoundError("ncu not available")
        if which("nsys") is None:
            raise FileNotFoundError("nsys not available")
        run_nvidia(args.input, args.model, subdir)
        return

    else:
        if which("rocprof") is None:
            raise FileNotFoundError("rocprof not available")

        run_amd(args.input, args.model, subdir)
        return

def run_nvidia(input, model, subdir):
    proc = subprocess.run(["nsys profile --gpu-metrics-device=all -o " + subdir + "/nsys_" + model + " --force-overwrite true " + input], shell=True, check=True, stdout=subprocess.PIPE)

def run_amd(input, model, subdir):
    os.environ["LIBOMPTARGET_KERNEL_TRACE"] = "2"
    if model == "omp":
        proc = subprocess.run(["rocprof -o " + subdir + "/omp.csv --stats  --hsa-trace " + input], shell=True, check=True, stderr=sys.stderr.fileno())
        if input[0] == "/" or input[0] == "~":
            proc = subprocess.run(["" + input + " 1> " + subdir + "/omp_stdout 2> " + subdir + "/omp_stderr"], shell=True, check=True, stderr=sys.stderr.fileno())
        else:
            proc = subprocess.run(["./" + input + " 1> " + subdir + "/omp_stdout 2> " + subdir + "/omp_stderr"], shell=True, check=True, stderr=sys.stderr.fileno())

    else:
        proc = subprocess.run(["rocprof -o " + subdir + "/hip.csv --stats --hip-trace --hsa-trace " + input], shell=True, check=True, stdout=subprocess.PIPE)



if  __name__ == "__main__":
    main()


