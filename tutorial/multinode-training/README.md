# Multi-Node Training with HugeCTR

This tutorial explains how to perform multi-node training with HugeCTR. 

## For Existing Clusters with a Job Scheduler
If your cluster is already equipped with a job scheduler, such as SLURM,
refer to the instructions in [dcn2nodes](../../samples/dcn/README.md).

## For New Clusters
If you need to create a new cluster, follow the instructions outlined below.

### Requirements
* Your nodes must be connected to one another through a shared directory so that they can visit each other using a ssh-nopassword
* OpenMPI version 4.0 
* Docker version 19.03.8

### To Create a New Cluster:
1. Prepare a [DCN dataset](../../samples/dcn/README.md) and cater the dataset in the same directory for all the nodes.

2. Build a [docker image](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#build-hugectr-from-source) in each node.

   You can also build an image once and use `docker save`/`docker load` to distribute the same image to all the  nodes. A production docker image of HugeCTR is available in the NVIDIA container repository. To pull and launch this container, see [Getting Started with NGC](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html#installing-hugectr-using-ngc-containers).
  
3. Build HugeCTR with [multi-nodes training supported](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_core_features.html#multi-node-training) and copy it to the same shared directory in each node.

4. Configure the python script.

   The [dcn_2node_8gpu.py](../../samples/dcn/dcn_2node_8gpu.py) is using two 8-GPU nodes. You can change the `"gpu"` setting based on the environment that you're using, and then add hugectr lib into PYTHONPATH:
    ```bash
    export PYTHONPATH=../../build/lib/
    ```

5. Configure `run_multinode.sh`.
   
   The [run_multinode.sh](./run_multinode.sh) uses `mpirun` to start the built docker container in each node. To use `run_multinode.sh`, you must set the following variables:
   * **WORK_DIR**: Parent path where you put the `hugectr` executable and your JSON config file.
   * **TEST_CMD**: how to run python script
   * **DATASET**: Real dataset path.
   * **VOL_DATASET**: Dataset path as shown inside your docker container as a mapping from `DATASET`. HugeCTR only sees this path.
   * **IMAGENAME**: Name of your Docker image.

6. After you set all the variables properly, run `bash run_multinode.sh` to start the multi-node training.

### For Advanced Users
* MPI must use TCP to build connections among nodes. If there are existing multiple network names for a given host, you may not receive a response or encounter a crash. You may need to exclude some of these network names by setting "btl_tcp_if_exclude". The detailed setting is dependent upon your environment.
* You may also consider customizing the `mpirun` arguments to achieve better performance on your own system. For example, you can set **bind** to **none** or **sockets**. 
* If you are using DGX1 or DGXA100, you can configure InfiniBand (IB) devices by running `source ./config_DGX1.sh` or `source ./config_DGXA100.sh` respectively.
