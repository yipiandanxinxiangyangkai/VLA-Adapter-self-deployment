# 环境配置

## 建立conda环境

```bash
# Create and activate conda environment
conda create -n vla-adapter python=3.10 -y
conda activate vla-adapter
```

## 安装Pytorch

我这里用的cuda版本是在4090D显卡上配置，版本号cuda==11.8,如果不是的也可以自行配置合适

```bash
# 安装 PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

#pip install to download dependencies
git clone https://github.com/OpenHelix-Team/VLA-Adapter.git
cd VLA-Adapter-main
pip install -e .
```

## 安装Flash Attention 2

```bash
pip install packaging ninja
ninja --version; echo $?
pip install flash-attn==2.5.5 --no-build-isolation --no-cache-dir -U
```
如果直接执行上面的命令一直卡在Building wheel for flash-attn(setup.py)|，可以尝试如下方法解决

```bash
#1.从https://github.com/Dao-AILab/flash-attention/releases下载flash-attn编译好的版本并下载。
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

#2.将编译文件下载到本地之后安装文件
pip install flash_attn-2.5.5+cu122torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

```

# 下载和配置LIBERO数据集

本次训练集和测试集都是LIBERO。LIBERO 是 UT Austin 于 2023 年推出的机器人终身学习基准数据集，基于 RoboSuite 与 MuJoCo 物理引擎构建。它包含 130 个涵盖空间变化（Spatial）、物体变化（Object）、目标变化（Goal）及长程任务（Long）的仿真任务，旨在利用 MuJoCo 高精度的物理反馈与强大的 OpenGL 渲染能力，测试模型在复杂操作中的视觉泛化、指令理解及持续学习能力，通过动画直观展示VLA模型的性能。

## 安装LIBERO数据集的环境

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
pip install -r requirements.txt
cd ..
pip install -r experiments/robot/libero/libero_requirements.txt
```

## 下载训练所用RLDS格式的LIBERO数据集文件

```bash
cd ~/VLA-Adapter-main
# 建议开启学术加速后再执行
source /etc/network_turbo

#方式一：采用git命令下载
git clone https://hf-mirror.com/datasets/openvla/modified_libero_rlds

#方式二：采用Huggingface镜像下载
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openvla/modified_libero_rlds --local-dir data/
```

## 安装mujoco仿真的渲染文件

```bash
apt-get update
apt-get install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev
```

# 预训练模型下载

prism-qwen25-extra-dinosiglip-224px-0_5b 是由 Stanford ILIAD 实验室（OpenVLA 团队）发布的一个轻量级多模态视觉语言模型（VLM）骨干网络。它是专门为机器人具身智能和 VLA模型设计的。由于该模型过大，模型需要从[https://hf-mirror.com/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b](https://hf-mirror.com/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b).下载到pretrained_models文件夹（文件夹有配置好的configs文件），文件的结构是:


```bash
├── pretrained_models
· ├── configs
└── prism-qwen25-extra-dinosiglip-224px-0_5b
```
下载模型的命令如下

```bash
#使用hf镜像下载prism-qwen25-extra-dinosiglip-224px-0_5b模型到pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b文件夹
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b --local-dir /root/autodl-tmp/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b
```

# LoRA微调

之后就是微调过程了，可以运行bash run.sh启动。bash run.sh这个文件中是针对Libero-Spatial任务微调，对于Libero-Object、Libero-Goal、Libero-微调的我使用的是4090D显卡训练，显存为24GB如果有更高规格显卡（比如RTX5090乃至A100/A800）可以参考[https://github.com/OpenHelix-Team/VLA-Adapter/blob/main/README.md](https://github.com/OpenHelix-Team/VLA-Adapter/blob/main/README.md).配置

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 /root/VLA-Adapter-main/vla-scripts/finetune.py \\
\--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \\
\--config_file_path pretrained_models/configs \\
\--data_root_dir data \\
\--dataset_name modified_libero_rlds/libero_spatial_no_noops \\
\--run_root_dir outputs \\
\--use_film False \\
\--num_images_in_input 2 \\
\--use_proprio True \\
\--use_lora True \\
\--use_fz False \\
\--use_minivlm True \\
\--image_aug True \\
\--num_steps_before_decay 400000 \\
\--max_steps 400005 \\
\--save_freq 5000 \\
\--save_latest_checkpoint_only False \\
\--merge_lora_during_training True \\
\--batch_size 1 \\
\--grad_accumulation_steps 8 \\
\--learning_rate 2e-4 \\
\--lora_rank 64 \\
\--use_pro_version True
```

# 推理

通过推理可以测试训练模型的性能。不同任务微调模型需要在不同测试集上测试，将pretrained_checkpoint修改模型路径、task_suite_name中名称修改成对应任务数据集即可。本项目测试的数据集是LIBERO基准的LIBERO_Spatial、LIBERO_Object、LIBERO_goal、LIBERO_Long，之前的指令已经将这四个数据集下载。

## 文件修改

LIBERO数据集的测试路径有微量的问题，为保障测试成功，需要对VLA-Adapter-main/LIBERO/libero/libero/\__init_\_.py的如下部分进行改变

```bash
# This is a default path for localizing all the default bddl files
bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")
# This is a default path for localizing all the default bddl files
init_states_default_path = os.path.join(benchmark_root_path, "./init_files")
# This is a default path for localizing all the default datasets
dataset_default_path = os.path.join(benchmark_root_path, "../datasets")
# This is a default path for localizing all the default assets
assets_default_path = os.path.join(benchmark_root_path, "./assets")
```

改成这样

```bash
# This is a default path for localizing all the default bddl files
bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")
# This is a default path for localizing all the default bddl files
init_states_default_path = os.path.join(benchmark_root_path, "./init_files")
# This is a default path for localizing all the default datasets
 #测试哪个数据集就把路径指向哪个数据集的测试集，比如测试LIBERO_Spatial数据集就把路径指向./init_files/libero_spatial，其他数据集 相应更换
dataset_default_path = os.path.join(benchmark_root_path, "./init_files/libero_spatial")
\# This is a default path for localizing all the default assets
assets_default_path = os.path.join(benchmark_root_path, "./assets")

```



## 开始推理


```bash
#LIBERO_Spatial数据集推理
PYTHONPATH=.:$(pwd)/LIBERO CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \\ #运行路径下的推理文件
\--num_trials_per_task 2 \\
\--use_proprio True \\
\--num_images_in_input 2 \\
\--use_film False \\
\--pretrained_checkpoint outputs/LIBERO-Spatial \\ #这个地方的路径是进行微调过的大语言模型路径，在outputs文件夹下，可将记录训练参数的长串模型名字改为以数据集命名的简短名字，比如LIBERO-Spatial数据集上微调的模型被命名为LIBERO-Spatial。为保证任务的适配性，不同任务需要对应微调数据集推理。
\--task_suite_name libero_spatial \\ #测试数据集指定
\--use_pro_version False \\

#LIBERO_Object数据集推理
PYTHONPATH=.:$(pwd)/LIBERO CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \\ #运行路径下的推理文件
\--num_trials_per_task 2 \\
\--use_proprio True \\
\--num_images_in_input 2 \\
\--use_film False \\
\--pretrained_checkpoint outputs/LIBERO-Object \\ #LIBERO-Object数据集上微调的模型
\--task_suite_name libero_object \\
\--use_pro_version False \\

#LIBERO_Goal数据集推理
PYTHONPATH=.:$(pwd)/LIBERO CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \\ #运行路径下的推理文件
\--num_trials_per_task 2 \\
\--use_proprio True \\
\--num_images_in_input 2 \\
\--use_film False \\
\--pretrained_checkpoint outputs/LIBERO-Goal \\ #LIBERO-Goal数据集上微调的模型
\--task_suite_name libero_object \\
\--use_pro_version False \\

#LIBERO_Long数据集推理
PYTHONPATH=.:$(pwd)/LIBERO CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \\ #运行路径下的推理文件
\--num_trials_per_task 2 \\
\--use_proprio True \\
\--num_images_in_input 2 \\
\--use_film False \\
\--pretrained_checkpoint outputs/LIBERO-10\\ #LIBERO-90数据集上微调的模型
\--task_suite_name libero_object \\
\--use_pro_version False \\
```