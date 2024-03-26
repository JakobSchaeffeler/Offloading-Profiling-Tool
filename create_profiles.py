import sys
import os
import subprocess
from shutil import which
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import re
#from prettytable import PrettyTable

def main():

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Processing of traces to uniform view')

    # Add optional argument --gpu with choices of nvidia or amd
    parser.add_argument('--gpu', choices=['nvidia', 'amd'], default="nvidia", help='Optional argument specifying GPU type (nvidia or amd)')

    # Add optional argument -o followed by a filename
    parser.add_argument('-o', metavar='OUTPUT_FILENAME', default="metrics", nargs=1, help='Optional subdir argument to store traces, default = metrics')

    # Optional Argument to choose offloading programming model, default omp
    parser.add_argument('--model', choices=['omp', 'hip', 'cuda'], default='omp', help='Optional argument specifying model (omp, hip, or cuda)')


    # Add non-optional argument for input dir
    parser.add_argument('subdir', metavar='SUBDIR_FILENAME', help='Non-optional argument specifying subdir where traces are located')

    args = parser.parse_args()
    out_dir = args.o[0]
    if not os.path.exists(out_dir):
           os.makedirs(out_dir)

    if args.gpu == "nvidia":
        if which("nsys") is None:
            raise FileNotFoundError("nsys not available")
        nvidia_handler(args.subdir, args.model, out_dir)
        return

    if args.gpu == "amd":
        amd_handler(args.subdir, args.model, out_dir)
        return

    print("FAILED")

# helper to convert elements in dic to floats if numeric
def convert_elems_to_float(dic):
    for elem in dic:
        for x in elem:
            if elem[x].replace('.','',1).replace(",",'',1).isdigit():
                elem[x] = float(elem[x].replace(",",'',1))

# helper to read csv file into dict
def read_csv_to_dict(path):
    dic = None
    with open(path) as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader)
        dic = [dict(zip(header, row)) for row in reader]
    return dic

# helper to read json file into dict
def read_json_to_dicts(path):
    js = open(path)
    dicts = json.load(js)
    return dicts

# helper to convert numeric strings in array to floats
def to_float_if_not(arr):
    for i in range(len(arr)):
        if not type(arr[i]) == float:
            arr[i] = float(arr[i].replace(',',''))


def amd_handler(subdir, model, out_dir):
        print(subdir)
        mem_dic = get_mem_stats_amd(subdir,model)
        dic = get_kernel_stats_amd(subdir, model)
        for d in dic:
            res_dic = {**dic[d], **mem_dic}
            res_dic["Model"] = model
            with open(out_dir + "/" + d + ".csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Value"])
                for k,v in res_dic.items():
                    writer.writerow([k,v])

        return

def nvidia_handler(subdir, model, out_dir):
        create_nvidia_profiles(subdir, model)
        mem_dic = get_mem_stats_nvidia(subdir)
        dic = get_kernel_stats_nvidia(mem_dic, subdir, model)
        for d in dic:
            res_dic = {**dic[d], **mem_dic}
            res_dic["Model"] = model
            with open(out_dir + "/" + d + ".csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Value"])
                for k,v in res_dic.items():
                    writer.writerow([k,v])

        return


# Create all nsys reports needed to extract the wanted metrics
def create_nvidia_profiles(subdir, model):
    ft = ".nsys-rep"
    if os.path.exists(subdir + "/nsys_" + model + ".qdrep"):
        ft = ".qdrep"
    print("nsys stats --report cuda_gpu_trace -o  " + subdir + "/nsys_gpu_trace --force-overwrite true --force-     export=true " + subdir + "/nsys_" + model +  ft)
    proc = subprocess.run(["nsys stats --report cuda_gpu_trace -o  " + subdir + "/nsys_gpu_trace --force-overwrite true --force-export=true " + subdir + "/nsys_" + model +  ft], shell=True, check=True, stdout=subprocess.PIPE)
    proc = subprocess.run(["nsys stats --report cuda_api_trace -o  " + subdir + "/nsys_api_trace --force-overwrite true --force-export=true " + subdir + "/nsys_" + model +  ft], shell=True, check=True, stdout=subprocess.PIPE)

    # get cuda_gpu_mem_time_sum
    proc = subprocess.run(["nsys stats --report cuda_gpu_mem_time_sum -o  " + subdir + "/nsys --force-overwrite true --force-export=true " + subdir + "/nsys_" + model +  ft], shell=True, check=True, stdout=subprocess.PIPE)

    # get cuda_gpu_mem_size_sum
    proc = subprocess.run(["nsys stats --report cuda_gpu_mem_size_sum -o  " + subdir + "/nsys --force-overwrite true --force-export=true " + subdir + "/nsys_" + model + ft], shell=True, check=True, stdout=subprocess.PIPE)

    # get  cuda_gpu_kern_sum
    proc = subprocess.run(["nsys stats --report cuda_gpu_kern_sum -o  " + subdir + "/nsys --force-overwrite true --force-export=true " + subdir + "/nsys_" + model + ft], shell=True, check=True, stdout=subprocess.PIPE)


    # get  cuda_gpu_kern_gb_sum
    proc = subprocess.run(["nsys stats --report cuda_gpu_kern_gb_sum -o  " + subdir + "/nsys --force-overwrite true --force-export=true " + subdir + "/nsys_" + model + ft], shell=True, check=True, stdout=subprocess.PIPE)


def get_mem_stats_nvidia(subdir):

    mem_time_sum = read_csv_to_dict(subdir + "/nsys_cuda_gpu_mem_time_sum.csv")
    convert_elems_to_float(mem_time_sum)

    mem_size_sum = read_csv_to_dict(subdir + "/nsys_cuda_gpu_mem_size_sum.csv")
    convert_elems_to_float(mem_size_sum)

    # extract DtoH values
    dtoh_time = list(filter(lambda elem: elem['Operation'] == '[CUDA memcpy DtoH]' or elem['Operation'] =='[CUDA memcpy Device-to-Host]', mem_time_sum))
    dtoh_time = dtoh_time[0] if len(dtoh_time) > 0 else {'Count':0, 'Total (MB)':0, 'Total Time (ns)':0}

    dtoh_size = list(filter(lambda elem: elem['Operation'] == '[CUDA memcpy DtoH]' or elem['Operation'] =='[CUDA memcpy Device-to-Host]', mem_size_sum))
    dtoh_size = dtoh_size[0] if len(dtoh_size) > 0 else {'Count':0, 'Total (MB)':0, 'Total Time (ns)':0}

    #count_dtoh = dtoh_time['Count']

    size_dtoh = dtoh_size['Total (MB)']

    time_dtoh = dtoh_time['Total Time (ns)']

    # extract HtoD values

    htod_time = list(filter(lambda elem: elem['Operation'] == '[CUDA memcpy HtoD]' or elem['Operation'] =='[CUDA memcpy Host-to-Device]', mem_time_sum))
    htod_time = htod_time[0] if len(htod_time) > 0 else {'Count':0, 'Total (MB)':0, 'Total Time (ns)':0}

    htod_size = list(filter(lambda elem: elem['Operation'] == '[CUDA memcpy HtoD]' or elem['Operation'] =='[CUDA memcpy Host-to-Device]', mem_size_sum))
    htod_size = htod_size[0] if len(htod_size) > 0 else {'Count':0, 'Total (MB)':0, 'Total Time (ns)':0}

    #count_htod = htod_time['Count']

    size_htod = htod_size['Total (MB)']

    time_htod = htod_time['Total Time (ns)']

    dic = {"HtoD Size": size_htod, "HtoD Time": time_htod, "DtoH Size": size_dtoh, "DtoH Time": time_dtoh}
    return dic


def get_kernel_stats_nvidia(stats_dic, subdir, model):
    kernsum = read_csv_to_dict(subdir + "/nsys_cuda_gpu_kern_sum.csv")
    convert_elems_to_float(kernsum)

    trace = read_csv_to_dict(subdir + "/nsys_gpu_trace_cuda_gpu_trace.csv")
    convert_elems_to_float(trace)

    kern_stats = read_csv_to_dict(subdir + "/nsys_cuda_gpu_kern_sum.csv")
    convert_elems_to_float(kern_stats)


    #Avg,Min,Max,Sync Avg, Sync Min, Sync Max, Calls, Total, Avg wgr, Min wgr, Max wgr, Avg grd, Min grd, Max grd
    dic = {}
    for d in kernsum:
        dic[d["Name"]] = {"Avg": d["Avg (ns)"], "Min": d["Min (ns)"], "Max": d["Max (ns)"], "Total": d["Total Time (ns)"], "Calls": d["Instances"]}
    dic = sorted(list(dic.items()), key=lambda k_v: k_v[1]['Avg'], reverse=True)
    dic = dict(dic)
    sync_times = traces_to_dict_nvidia(subdir,model)
    for name in sync_times.keys():
        dic[name]["Sync Min"] = np.min(sync_times[name]) if np.min(sync_times[name]) > 0 else dic[name]["Min"]
        dic[name]["Sync Max"] = np.max(sync_times[name]) if np.max(sync_times[name]) > 0 else dic[name]["Max"]
        dic[name]["Sync Avg"] = np.average(sync_times[name]) if np.average(sync_times[name]) > 0 else dic[name]["Avg"]
        dic[name]["Sync Total"] = np.sum(sync_times[name]) if np.sum(sync_times[name]) > 0 else dic[name]["Total"]

    thread_stats = read_csv_to_dict(subdir + "/nsys_cuda_gpu_kern_gb_sum.csv")
    #print(sync_times.keys())

    for name in sync_times.keys():
        for line in thread_stats:
            if name in line["Name"]:
                xyz = line["GridXYZ"].split()
                grid_size = int(xyz[0]) * int(xyz[1]) * int(xyz[2])

                xyz = line["BlockXYZ"].split()
                block_size = int(xyz[0]) * int(xyz[1]) * int(xyz[2])

                dic[name]["#Teams"] = grid_size
                dic[name]["#Threads"] = block_size
                break
    return dic


def traces_to_dict_nvidia(subdir, model):
    api_trace = read_csv_to_dict(subdir + "/nsys_api_trace_cuda_api_trace.csv")
    convert_elems_to_float(api_trace)
    gpu_trace = read_csv_to_dict(subdir + "/nsys_gpu_trace_cuda_gpu_trace.csv")
    convert_elems_to_float(gpu_trace)
    kern_sum = read_csv_to_dict(subdir + "/nsys_cuda_gpu_kern_sum.csv")

    max_j = 5 if model == "omp" else 1

    kernel_list = []
    kernel_names = []

    # list of kernel names
    for line in kern_sum:
        kernel_names.append(line["Name"])

    #get list of kernel calls in order
    for line in gpu_trace:
        if line["Name"] in kernel_names:
            kernel_list.append(line["Name"])

    sync_times = []

    for i in range(len(api_trace)):
        line = api_trace[i]
        if "LaunchKernel" in line["Name"]:
            start_time = line["Start (ns)"]
            i += 1
            line = api_trace[i]
            j = 0
            while not "Synchronize" in line["Name"]: #TODO: really want time to next sync?
                i+=1
                j+=1
                line = api_trace[i]
                if j > max_j:
                    break
            time = 0
            if j > max_j:
                time = 0
            else:
                sync = line["Duration (ns)"] + line["Start (ns)"]
                time = sync - start_time
            i += 1
            sync_times.append(time)
    dicts = {}
    for i in range(len(kernel_list)):
        kernel = kernel_list[i]
        if dicts.get(kernel) is None:
            dicts[kernel] = [sync_times[i]]
        else:
            dicts[kernel].append(sync_times[i])
    return dicts

def get_bytes_copied_amd(dicts):
    bytes_htod = 0
    bytes_dtoh = 0
    count_htod = 0
    count_dtoh = 0
    time_htod = 0
    time_dtoh = 0
    for dic in dicts["traceEvents"]:
        if dic.get("args") is not None:
            if dic["args"].get("Name") == "hipMemcpy":

                args = dic["args"]["args"]
                res = re.match(r"^.*sizeBytes\((.*)\) .*$", dic["args"]["args"])
                byte = int(res.group(1))
                kind = re.match(r"^.*kind\((.*)\)\).*$", dic["args"]["args"])
                kind = int(kind.group(1))
                time = int(dic["args"]["DurationNs"])
                if kind == 1:
                    bytes_htod += byte
                    count_htod += 1
                    time_htod += time
                elif kind == 2:
                    bytes_dtoh += byte
                    count_dtoh += 1
                    time_dtoh += time
    return bytes_htod, bytes_dtoh, count_htod, count_dtoh, time_htod, time_dtoh

def get_omp_mem_stats_amd(subdir):
    stderr = open(subdir + "/omp_stderr")
    stderr = stderr.readlines()
    bytes_htod = 0
    bytes_dtoh = 0
    count_htod = 0
    count_dtoh = 0
    time_htod = 0
    time_dtoh = 0

    for i in range(len(stderr)):
        line = stderr[i]
        if "__tgt_rtl_data_retrieve_async" in line:
            time = re.split(r' +', line)[2]
            time = float(re.sub('[^0-9]','', time))
            byte = re.split(r' +', line)[8]
            byte = int(re.sub('[^0-9]','', byte))
            i += 1
            count_dtoh += 1
            bytes_dtoh += byte
            line = stderr[i]
            if "__tgt_rtl_synchronize" in line:
                sync = re.split(r' +', line)[2]
                sync = float(re.sub('[^0-9]','', sync))
                time += sync
                i += 1
                line = stderr[i]
            time_dtoh += time * 1000

    for i in range(len(stderr)):
        line = stderr[i]
        if "__tgt_rtl_data_submit_async" in line:
            time = re.split(r' +', line)[2]
            time = float(re.sub('[^0-9]','', time))
            byte = re.split(r' +', line)[8]
            byte = int(re.sub('[^0-9]','', byte))
            i += 1
            count_htod += 1
            bytes_htod += byte
            line = stderr[i]
            if "__tgt_rtl_synchronize" in line:
                sync = re.split(r' +', line)[2]
                sync = float(re.sub('[^0-9]','', sync))
                time += sync
                i += 1
                line = stderr[i]
            time_htod += time * 1000
    return bytes_htod, bytes_dtoh, count_htod, count_dtoh, time_htod, time_dtoh




def get_mem_stats_amd(subdir, model):
    # stats: #Calls, #Bytes transfered, Total Time
    size_dtoh = 0
    size_htod = 0
    count_dtoh = 0
    count_htod = 0
    time_dtoh = 0
    time_htod = 0
    if model == "omp":
        size_htod, size_dtoh, count_htod, count_dtoh, time_htod, time_dtoh  = get_omp_mem_stats_amd(subdir)
    else:
        trace_dicts = read_json_to_dicts(subdir + "/hip.json")
        size_htod, size_dtoh, count_htod, count_dtoh, time_htod, time_dtoh = get_bytes_copied_amd(trace_dicts)


    dic = {"HtoD Size": size_htod/1000000, "HtoD Time": time_htod, "DtoH Size": size_dtoh/1000000, "DtoH Time": time_dtoh}
    return dic




def create_dict_of_kernels(sum_dict):
    kernel_dict = {}
    for row in sum_dict:
        if row["KernelName"] in kernel_dict:
            kernel_dict[row["KernelName"]]['Time'].append(row["DurationNs"])
            kernel_dict[row["KernelName"]]['grd'].append(row["grd"])
            kernel_dict[row["KernelName"]]['wgr'].append(row["wgr"])
        else:
            kernel_dict[row["KernelName"]] = {'Time': [row["DurationNs"]], 'grd': [row["grd"]], 'wgr': [row["wgr"]]}
    return kernel_dict

def compute_amd_averages(kernels_dict):
    for name in kernels_dict:
        elem = kernels_dict[name]
        time = np.array(elem['Time'])

        elem['Avg'] = np.average(time)
        elem['Min'] = np.min(time)
        elem['Max'] = np.max(time)
        wgr = np.array(elem['wgr'])
        elem['#Threads'] = np.average(wgr)
        #elem['Min wgr'] = np.min(wgr)
        #elem['Max wgr'] = np.max(wgr)
        grd = np.array(elem['grd'])
        elem['#Teams'] = np.average(grd)/elem['#Threads']
        elem.pop("grd")
        elem.pop("wgr")
        elem.pop("Time")
        #elem['Min grd'] = np.min(grd)/elem['Min wgr']
        #elem['Max grd'] = np.max(grd)/elem['Max wgr']
        #elem['Calls'] = len(elem['Time'])
        #elem['Total'] = np.sum(time)

def shorten_omp_names(namelist):
    for i in range(len(namelist)):
        if "omp_offloading" in namelist[i]:
            namelist[i] = "omp_offloading" +  namelist[i].split("omp_offloading_")[-1] #("__omp_offloading_33_40287c3__ZN9","",1)
    return namelist


def omp_traces_to_dict_amd(subdir):
    stdout = open(subdir + "/omp_stdout")
    stderr = open(subdir + "/omp_stderr")
    stdout = stdout.readlines()
    stderr = stderr.readlines()

    kernel_list = []
    for line in stdout:
        if "__omp_offloading_" in line:
            start = line.find("__omp_offloading_")
            kernel_list.append(line[start:-1])

    sync_times = []
    for i in range(len(stderr)):
        line = stderr[i]
        if "__tgt_rtl_run_target_team_region_async" in line:
            time = re.split(r' +', line)[2]
            time = float(re.sub('[^0-9]','', time))
            i += 1
            line = stderr[i]
            while not "__tgt_rtl_synchronize" in line: #TODO: really want time to next sync?
                sync = re.split(r' +', line)[2]
                sync = float(re.sub('[^0-9]','', sync))
                time += sync
                i += 1
                line = stderr[i]
            sync = re.split(r' +', line)[2]
            sync = float(re.sub('[^0-9]','', sync))
            time += sync
            i += 1
            sync_times.append(time)
    dicts = {}
    for i in range(len(kernel_list)):
        kernel = kernel_list[i]
        if dicts.get(kernel) is None:
            dicts[kernel] = [sync_times[i]]
        else:
            dicts[kernel].append(sync_times[i])
    return dicts


def get_kernel_stats_amd(subdir, model):
    kernel_time_sum = None
    if model == "hip":
        kernel_time_sum = read_csv_to_dict(subdir + "/hip.csv")
    else:
        kernel_time_sum = read_csv_to_dict(subdir + "/omp.csv")

    convert_elems_to_float(kernel_time_sum)
    kernels_dict = create_dict_of_kernels(kernel_time_sum)
    compute_amd_averages(kernels_dict)
    kernels_dict = sorted(list(kernels_dict.items()), key=lambda k_v: k_v[1]['Avg'], reverse=True)
    kernels_dict = dict(kernels_dict)

    if model == "omp":
        omp_sync_times = omp_traces_to_dict_amd(subdir)
        tmp_dict = {}
        for name in omp_sync_times.keys():
            for dn in kernels_dict.keys():
                if name in dn:
                    tmp_dict[name] = kernels_dict[dn]
                    tmp_dict[name]["Sync Avg"] = np.average(omp_sync_times[name]) * 1000
                    tmp_dict[name]["Sync Min"] = np.min(omp_sync_times[name]) * 1000
                    tmp_dict[name]["Sync Max"] = np.max(omp_sync_times[name]) * 1000
                    #tmp_dict[name]["Sync Time"] = np.array(omp_sync_times[name]) * 1000
                    #tmp_dict[name]["Sync Total"] = np.sum(omp_sync_times[name]) * 1000
        kernels_dict = dict(sorted(list(tmp_dict.items()), key=lambda k_v: k_v[1]['Avg'], reverse=True))
    else:
        for name in kernels_dict.keys():
                kernels_dict[name]["Sync Avg"] = kernels_dict[name]["Avg"]
                kernels_dict[name]["Sync Min"] = kernels_dict[name]["Min"]
                kernels_dict[name]["Sync Max"] = kernels_dict[name]["Max"]

    return kernels_dict


if  __name__ == "__main__":
    main()


