<center>

# Harmony in Diversity: Improving All-in-One Image Restoration via Multi-Task Collaboration


</center>

## Overview
>Deep learning-based all-in-one image restoration methods have garnered significant attention in recent years due to capable of addressing multiple degradation tasks. These methods focus on extracting task-oriented information to guide the unified model and have achieved promising results through elaborate architecture design. They commonly adopt a simple mix training paradigm, and the proper optimization strategy for all-in-one tasks has been scarcely investigated. This oversight neglects the intricate relationships and potential conflicts among various restoration tasks, consequently leading to inconsistent optimization rhythms.
In this paper, we extend and redefine the conventional all-in-one image restoration task as a multi-task learning problem and propose a straightforward yet effective active-reweighting strategy, dubbed $\textbf{Art}$, to harmonize the optimization of multiple degradation tasks. Art is a plug-and-play optimization strategy designed to mitigate hidden conflicts among multi-task optimization processes.
Through extensive experiments on a diverse range of all-in-one image restoration settings, Art has been demonstrated to substantially enhance the performance of existing methods. When incorporated into the AirNet and TransWeather models, it achieves average improvements of $\textbf{1.16}$ dB and $\textbf{1.24}$ dB on PSNR, respectively. We hope this work will provide a principled framework for collaborating multiple tasks in all-in-one image restoration and pave the way for more efficient and effective restoration models, ultimately advancing the state-of-the-art in this critical research domain.

<a href="https://www.imagehub.cc/image/bd9oNr"><img src="https://s1.imagehub.cc/images/2024/04/18/605847469e2604f305018f19d9b49c7d.md.png" alt="605847469e2604f305018f19d9b49c7d.png" border="0" /></a>


## Results

<a href="https://www.imagehub.cc/image/bd99bs"><img src="https://s1.imagehub.cc/images/2024/04/18/5f3781c55a98489cc2aee723be03b57c.md.png" alt="5f3781c55a98489cc2aee723be03b57c.png" border="0" /></a>



Visual results of test datasets of our retrained AirNet and TransWeather are available at [BaiduPan](https://pan.baidu.com/s/12kYrodXUQInL8mPfWfPLQw?pwd=artr) with password artr




## Reproduce

Following [Transweather](https://github.com/jeya-maria-jose/TransWeather), download AllWeather dataset and change the path to datasets in `transweather.yml`

```
python basicsr/train.py -opt options/train/Transweather/Transweather.yml 
```
