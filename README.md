In this repository, I'll start a project to show how to build a Person ReID.

```
服务器文件树
.
├── baseline                                                ：存放baseline阶段训练文件
│   ├── config.py                                           ：模型配置文件
│   ├── final_logs                                          ：各个优化阶段模型log
│   │   ├── baseline_model_202005100000.h5
│   │   │   └── events.out.tfevents.1588924014.gpu
│   │   ├── baseline_model_202005131024.h5
│   │   │   └── events.out.tfevents.1589336697.gpu
│   │   └── baseline_model_202005141719.h5
│   │       └── events.out.tfevents.1589448003.gpu
│   ├── nohup.out                                           ：后台输出文件
│   ├── __pycache__
│   │   ├── config.cpython-36.pyc
│   │   └── utils.cpython-36.pyc
│   ├── TB_nohup.txt
│   ├── train_baseline_cnn.py                               ：模型训练文件
│   ├── train_cnn_baseline.py                               ：Seven提供框架
│   └── utils.py                                            ：模型工具文件
├── database                                                ：数据集（压缩好的数据集已存放在/data/PersonReID/market1501）
│   └── Market-1501-v15.09.15.zip
├── MobileNetV2                                             ：backbone模型
│   ├── mobilenet_1_0_224_tf_no_top.h5
│   └── mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_96_no_top.h5
├── models                                                  ：checkpoint 模型权重
│   ├── baseline_model_202005100000.h5
│   ├── baseline_model_202005131024.h5
│   ├── baseline_model_202005141536.h5
│   ├── baseline_model_202005141609.h5
│   ├── baseline_model_202005141717.h5
│   └── baseline_model_202005141719.h5
├── README.md                                               ：说明
└── ReID项目问题记录.xmind                                   ：各阶段实验报告

9 directories, 22 files
```
修改
