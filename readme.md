# MMSD

### DownLoad Dataset：
 - 链接: https://caiyun.139.com/m/i?085CtKTgnhbl7  提取码:xmSn  

### Official Materials
 - TF:  https://github.com/QinYi-team/MMSD/tree/main
 - Paper： 
   - [PUBLISHED PAPERS](https://doi.org/10.1016/j.knosys.2023.110748)
   
     ​     
   

### Cited:
```html
@article{QIAN2023110748,
title = {Maximum mean square discrepancy: A new discrepancy representation metric for mechanical fault transfer diagnosis},
journal = {Knowledge-Based Systems},
pages = {110748},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.110748},
url = {https://www.sciencedirect.com/science/article/pii/S0950705123004987},
author = {Quan Qian and Yi Wang and Taisheng Zhang and Yi Qin},
keywords = {Discrepancy representation metric, Fault transfer diagnosis, Unsupervised domain adaptation, Planetary gearbox},
abstract = {Discrepancy representation metric completely determines the transfer diagnosis performance of deep domain adaptation methods. Maximum mean discrepancy (MMD) based on the mean statistic, as the commonly used metric, has poor discrepancy representation in some cases. MMD is generally known from the aspect of kernel function, but the inherent relationship between the two is unknown. To deal with these issues, the authors theoretically explore their relationship first. With the revealed relationship, a novel discrepancy representation metric named maximum mean square discrepancy (MMSD), which can comprehensively reflect the mean and variance information of data samples in the reproducing kernel Hilbert space, is constructed for enhancing domain confusion. Additionally, for the real application under limited samples and ensuring the effectiveness of MMSD, biased and unbiased empirical MMSD statistics are developed, and the error bounds between the two empirical statistics and the real distribution discrepancy are derived. The proposed MMSD is successfully applied to the end-to-end fault diagnosis of planetary gearbox of wind turbine without labeled target-domain samples. The experimental results on twelve cross-load transfer tasks validate that MMSD has a better ability of discrepancy representation and a higher diagnosis accuracy compared with other well-known discrepancy representation metrics. The related code can be downloaded from https://qinyi-team.github.io/#blog.}
```
### Experience

1. 平滑交叉熵损失要比交叉熵损失的效果更好。
2. 使用lambd，使得迁移的效果更加稳定。

### Usage

建议大家将开源的MMSD域差异度量损失（即插即用）在赵老师的开源的代码中替换掉度量模块。(https://github.com/ZhaoZhibin/UDTL)

替换方式：

```python
from MMSD import MMSD
criterion2 = MMSD()
loss = criterion2(output1, output2)
```

### Experimental result

```python
Epoch1, test_loss is 1.70728, train_accuracy is 0.96575,test_accuracy is 0.21875,train_all_loss is 0.47141,source_cla_loss is 0.47141,domain_loss is 0.11647
Epoch2, test_loss is 2.35181, train_accuracy is 1.00000,test_accuracy is 0.27100,train_all_loss is 0.37312,source_cla_loss is 0.35756,domain_loss is 0.05892
Epoch3, test_loss is 2.27180, train_accuracy is 1.00000,test_accuracy is 0.17525,train_all_loss is 0.36413,source_cla_loss is 0.35554,domain_loss is 0.02357
Epoch4, test_loss is 2.52417, train_accuracy is 1.00000,test_accuracy is 0.16125,train_all_loss is 0.36036,source_cla_loss is 0.35355,domain_loss is 0.01553
Epoch5, test_loss is 2.35286, train_accuracy is 1.00000,test_accuracy is 0.23850,train_all_loss is 0.35704,source_cla_loss is 0.35188,domain_loss is 0.01035
Epoch6, test_loss is 2.11720, train_accuracy is 1.00000,test_accuracy is 0.40925,train_all_loss is 0.35786,source_cla_loss is 0.35186,domain_loss is 0.01090
Epoch7, test_loss is 2.10492, train_accuracy is 1.00000,test_accuracy is 0.33450,train_all_loss is 0.35841,source_cla_loss is 0.35179,domain_loss is 0.01108
Epoch8, test_loss is 2.28453, train_accuracy is 1.00000,test_accuracy is 0.28400,train_all_loss is 0.35728,source_cla_loss is 0.35096,domain_loss is 0.00990
Epoch9, test_loss is 2.61733, train_accuracy is 1.00000,test_accuracy is 0.25325,train_all_loss is 0.35661,source_cla_loss is 0.35092,domain_loss is 0.00840
Epoch10, test_loss is 2.60611, train_accuracy is 1.00000,test_accuracy is 0.11200,train_all_loss is 0.36039,source_cla_loss is 0.35201,domain_loss is 0.01178
Epoch11, test_loss is 2.20323, train_accuracy is 1.00000,test_accuracy is 0.43500,train_all_loss is 0.36111,source_cla_loss is 0.35251,domain_loss is 0.01155
Epoch12, test_loss is 2.52880, train_accuracy is 1.00000,test_accuracy is 0.26050,train_all_loss is 0.35741,source_cla_loss is 0.35055,domain_loss is 0.00884
Epoch13, test_loss is 2.43795, train_accuracy is 1.00000,test_accuracy is 0.31250,train_all_loss is 0.35709,source_cla_loss is 0.35001,domain_loss is 0.00879
Epoch14, test_loss is 2.20165, train_accuracy is 1.00000,test_accuracy is 0.30825,train_all_loss is 0.35859,source_cla_loss is 0.35034,domain_loss is 0.00990
Epoch15, test_loss is 2.32735, train_accuracy is 1.00000,test_accuracy is 0.32250,train_all_loss is 0.35892,source_cla_loss is 0.35087,domain_loss is 0.00936
Epoch16, test_loss is 1.98427, train_accuracy is 1.00000,test_accuracy is 0.41350,train_all_loss is 0.35704,source_cla_loss is 0.35041,domain_loss is 0.00749
Epoch17, test_loss is 2.40874, train_accuracy is 1.00000,test_accuracy is 0.26625,train_all_loss is 0.35763,source_cla_loss is 0.35074,domain_loss is 0.00758
Epoch18, test_loss is 1.98695, train_accuracy is 1.00000,test_accuracy is 0.43575,train_all_loss is 0.35825,source_cla_loss is 0.35019,domain_loss is 0.00865
Epoch19, test_loss is 2.52306, train_accuracy is 1.00000,test_accuracy is 0.19475,train_all_loss is 0.35888,source_cla_loss is 0.34970,domain_loss is 0.00962
Epoch20, test_loss is 1.95578, train_accuracy is 0.99975,test_accuracy is 0.48125,train_all_loss is 0.35809,source_cla_loss is 0.35103,domain_loss is 0.00723
Epoch21, test_loss is 2.57803, train_accuracy is 1.00000,test_accuracy is 0.30950,train_all_loss is 0.35776,source_cla_loss is 0.35048,domain_loss is 0.00730
Epoch22, test_loss is 2.40677, train_accuracy is 1.00000,test_accuracy is 0.32975,train_all_loss is 0.35670,source_cla_loss is 0.35015,domain_loss is 0.00644
Epoch23, test_loss is 2.32670, train_accuracy is 1.00000,test_accuracy is 0.37975,train_all_loss is 0.35628,source_cla_loss is 0.35001,domain_loss is 0.00604
Epoch24, test_loss is 2.42710, train_accuracy is 1.00000,test_accuracy is 0.29125,train_all_loss is 0.35750,source_cla_loss is 0.34973,domain_loss is 0.00735
Epoch25, test_loss is 2.31670, train_accuracy is 1.00000,test_accuracy is 0.35600,train_all_loss is 0.35724,source_cla_loss is 0.34979,domain_loss is 0.00692
Epoch26, test_loss is 2.13007, train_accuracy is 1.00000,test_accuracy is 0.29450,train_all_loss is 0.35663,source_cla_loss is 0.34949,domain_loss is 0.00653
Epoch27, test_loss is 2.56636, train_accuracy is 1.00000,test_accuracy is 0.22075,train_all_loss is 0.35791,source_cla_loss is 0.34966,domain_loss is 0.00741
Epoch28, test_loss is 2.61027, train_accuracy is 1.00000,test_accuracy is 0.27125,train_all_loss is 0.35974,source_cla_loss is 0.35006,domain_loss is 0.00856
Epoch29, test_loss is 2.64836, train_accuracy is 1.00000,test_accuracy is 0.15350,train_all_loss is 0.36258,source_cla_loss is 0.35029,domain_loss is 0.01071
Epoch30, test_loss is 1.93399, train_accuracy is 1.00000,test_accuracy is 0.45050,train_all_loss is 0.36011,source_cla_loss is 0.35069,domain_loss is 0.00809
Epoch31, test_loss is 2.48554, train_accuracy is 1.00000,test_accuracy is 0.29175,train_all_loss is 0.35629,source_cla_loss is 0.34977,domain_loss is 0.00552
Epoch32, test_loss is 2.21559, train_accuracy is 1.00000,test_accuracy is 0.29150,train_all_loss is 0.35875,source_cla_loss is 0.34962,domain_loss is 0.00762
Epoch33, test_loss is 1.93778, train_accuracy is 1.00000,test_accuracy is 0.48025,train_all_loss is 0.36135,source_cla_loss is 0.35016,domain_loss is 0.00923
Epoch34, test_loss is 2.15657, train_accuracy is 0.99900,test_accuracy is 0.38750,train_all_loss is 0.36303,source_cla_loss is 0.35151,domain_loss is 0.00938
Epoch35, test_loss is 2.09539, train_accuracy is 1.00000,test_accuracy is 0.41875,train_all_loss is 0.35931,source_cla_loss is 0.35029,domain_loss is 0.00725
Epoch36, test_loss is 2.14911, train_accuracy is 1.00000,test_accuracy is 0.41450,train_all_loss is 0.35856,source_cla_loss is 0.34987,domain_loss is 0.00691
Epoch37, test_loss is 2.30350, train_accuracy is 1.00000,test_accuracy is 0.35275,train_all_loss is 0.35922,source_cla_loss is 0.34980,domain_loss is 0.00740
Epoch38, test_loss is 2.01832, train_accuracy is 1.00000,test_accuracy is 0.42125,train_all_loss is 0.36115,source_cla_loss is 0.34977,domain_loss is 0.00884
Epoch39, test_loss is 2.12754, train_accuracy is 1.00000,test_accuracy is 0.43925,train_all_loss is 0.36185,source_cla_loss is 0.35056,domain_loss is 0.00867
Epoch40, test_loss is 2.27010, train_accuracy is 1.00000,test_accuracy is 0.38275,train_all_loss is 0.35945,source_cla_loss is 0.34958,domain_loss is 0.00749
Epoch41, test_loss is 2.27428, train_accuracy is 1.00000,test_accuracy is 0.37300,train_all_loss is 0.35898,source_cla_loss is 0.34955,domain_loss is 0.00709
Epoch42, test_loss is 3.02612, train_accuracy is 1.00000,test_accuracy is 0.05425,train_all_loss is 0.36095,source_cla_loss is 0.35021,domain_loss is 0.00799
Epoch43, test_loss is 1.81688, train_accuracy is 0.99975,test_accuracy is 0.50675,train_all_loss is 0.36015,source_cla_loss is 0.35015,domain_loss is 0.00737
Epoch44, test_loss is 2.43221, train_accuracy is 1.00000,test_accuracy is 0.30950,train_all_loss is 0.35984,source_cla_loss is 0.34973,domain_loss is 0.00737
Epoch45, test_loss is 2.29550, train_accuracy is 1.00000,test_accuracy is 0.33425,train_all_loss is 0.35929,source_cla_loss is 0.34962,domain_loss is 0.00699
Epoch46, test_loss is 1.83346, train_accuracy is 1.00000,test_accuracy is 0.48475,train_all_loss is 0.36158,source_cla_loss is 0.34975,domain_loss is 0.00846
Epoch47, test_loss is 2.47635, train_accuracy is 0.99975,test_accuracy is 0.30550,train_all_loss is 0.36134,source_cla_loss is 0.35078,domain_loss is 0.00749
Epoch48, test_loss is 2.31410, train_accuracy is 1.00000,test_accuracy is 0.31875,train_all_loss is 0.36162,source_cla_loss is 0.35039,domain_loss is 0.00789
Epoch49, test_loss is 2.26891, train_accuracy is 1.00000,test_accuracy is 0.36725,train_all_loss is 0.35993,source_cla_loss is 0.34990,domain_loss is 0.00698
Epoch50, test_loss is 2.08352, train_accuracy is 1.00000,test_accuracy is 0.38875,train_all_loss is 0.36137,source_cla_loss is 0.34981,domain_loss is 0.00798
Epoch51, test_loss is 2.42994, train_accuracy is 0.99975,test_accuracy is 0.29925,train_all_loss is 0.35795,source_cla_loss is 0.34976,domain_loss is 0.00560
Epoch52, test_loss is 2.04134, train_accuracy is 0.99975,test_accuracy is 0.45800,train_all_loss is 0.36056,source_cla_loss is 0.35033,domain_loss is 0.00694
Epoch53, test_loss is 2.09900, train_accuracy is 0.99925,test_accuracy is 0.49200,train_all_loss is 0.36229,source_cla_loss is 0.35129,domain_loss is 0.00740
Epoch54, test_loss is 2.87855, train_accuracy is 1.00000,test_accuracy is 0.09150,train_all_loss is 0.36451,source_cla_loss is 0.35035,domain_loss is 0.00945
Epoch55, test_loss is 2.07624, train_accuracy is 1.00000,test_accuracy is 0.39200,train_all_loss is 0.35990,source_cla_loss is 0.34988,domain_loss is 0.00664
Epoch56, test_loss is 2.04308, train_accuracy is 1.00000,test_accuracy is 0.42925,train_all_loss is 0.35937,source_cla_loss is 0.34965,domain_loss is 0.00638
Epoch57, test_loss is 2.18549, train_accuracy is 1.00000,test_accuracy is 0.40975,train_all_loss is 0.35868,source_cla_loss is 0.34950,domain_loss is 0.00599
Epoch58, test_loss is 1.90264, train_accuracy is 1.00000,test_accuracy is 0.44225,train_all_loss is 0.35848,source_cla_loss is 0.34946,domain_loss is 0.00584
Epoch59, test_loss is 2.24085, train_accuracy is 0.99975,test_accuracy is 0.39500,train_all_loss is 0.36069,source_cla_loss is 0.35044,domain_loss is 0.00659
Epoch60, test_loss is 2.16645, train_accuracy is 1.00000,test_accuracy is 0.41550,train_all_loss is 0.35808,source_cla_loss is 0.34961,domain_loss is 0.00541
Epoch61, test_loss is 2.32162, train_accuracy is 1.00000,test_accuracy is 0.31100,train_all_loss is 0.36257,source_cla_loss is 0.35030,domain_loss is 0.00777
Epoch62, test_loss is 2.02297, train_accuracy is 1.00000,test_accuracy is 0.44625,train_all_loss is 0.35716,source_cla_loss is 0.34946,domain_loss is 0.00484
Epoch63, test_loss is 2.17477, train_accuracy is 1.00000,test_accuracy is 0.40525,train_all_loss is 0.36074,source_cla_loss is 0.34959,domain_loss is 0.00696
Epoch64, test_loss is 2.01394, train_accuracy is 1.00000,test_accuracy is 0.45350,train_all_loss is 0.35919,source_cla_loss is 0.34956,domain_loss is 0.00597
Epoch65, test_loss is 2.04567, train_accuracy is 1.00000,test_accuracy is 0.43150,train_all_loss is 0.35964,source_cla_loss is 0.34948,domain_loss is 0.00626
Epoch66, test_loss is 2.26130, train_accuracy is 1.00000,test_accuracy is 0.31925,train_all_loss is 0.35874,source_cla_loss is 0.34962,domain_loss is 0.00558
Epoch67, test_loss is 2.03899, train_accuracy is 1.00000,test_accuracy is 0.43050,train_all_loss is 0.36048,source_cla_loss is 0.34988,domain_loss is 0.00644
Epoch68, test_loss is 2.26282, train_accuracy is 1.00000,test_accuracy is 0.32625,train_all_loss is 0.35656,source_cla_loss is 0.34929,domain_loss is 0.00439
Epoch69, test_loss is 2.14964, train_accuracy is 1.00000,test_accuracy is 0.40100,train_all_loss is 0.36315,source_cla_loss is 0.34980,domain_loss is 0.00800
Epoch70, test_loss is 2.09022, train_accuracy is 1.00000,test_accuracy is 0.42075,train_all_loss is 0.35800,source_cla_loss is 0.34952,domain_loss is 0.00505
Epoch71, test_loss is 1.91966, train_accuracy is 1.00000,test_accuracy is 0.46025,train_all_loss is 0.35967,source_cla_loss is 0.34967,domain_loss is 0.00592
Epoch72, test_loss is 2.52028, train_accuracy is 1.00000,test_accuracy is 0.25550,train_all_loss is 0.36009,source_cla_loss is 0.35002,domain_loss is 0.00592
Epoch73, test_loss is 2.20228, train_accuracy is 1.00000,test_accuracy is 0.37700,train_all_loss is 0.35946,source_cla_loss is 0.34967,domain_loss is 0.00573
Epoch74, test_loss is 1.94529, train_accuracy is 0.99925,test_accuracy is 0.49575,train_all_loss is 0.36289,source_cla_loss is 0.35127,domain_loss is 0.00675
Epoch75, test_loss is 2.82669, train_accuracy is 0.99850,test_accuracy is 0.25000,train_all_loss is 0.36925,source_cla_loss is 0.35456,domain_loss is 0.00849
Epoch76, test_loss is 2.24564, train_accuracy is 0.99950,test_accuracy is 0.37200,train_all_loss is 0.36014,source_cla_loss is 0.35083,domain_loss is 0.00534
Epoch77, test_loss is 2.10572, train_accuracy is 1.00000,test_accuracy is 0.43425,train_all_loss is 0.35951,source_cla_loss is 0.34934,domain_loss is 0.00580
Epoch78, test_loss is 2.12904, train_accuracy is 1.00000,test_accuracy is 0.36225,train_all_loss is 0.36563,source_cla_loss is 0.34958,domain_loss is 0.00910
Epoch79, test_loss is 2.17708, train_accuracy is 1.00000,test_accuracy is 0.41100,train_all_loss is 0.36162,source_cla_loss is 0.35044,domain_loss is 0.00631
Epoch80, test_loss is 2.25490, train_accuracy is 1.00000,test_accuracy is 0.39525,train_all_loss is 0.35696,source_cla_loss is 0.34930,domain_loss is 0.00430
Epoch81, test_loss is 2.12288, train_accuracy is 1.00000,test_accuracy is 0.42675,train_all_loss is 0.36222,source_cla_loss is 0.34962,domain_loss is 0.00702
Epoch82, test_loss is 2.25930, train_accuracy is 1.00000,test_accuracy is 0.32775,train_all_loss is 0.36017,source_cla_loss is 0.34945,domain_loss is 0.00595
Epoch83, test_loss is 2.55101, train_accuracy is 1.00000,test_accuracy is 0.26325,train_all_loss is 0.36796,source_cla_loss is 0.35097,domain_loss is 0.00937
Epoch84, test_loss is 1.75571, train_accuracy is 0.99975,test_accuracy is 0.55050,train_all_loss is 0.36502,source_cla_loss is 0.35037,domain_loss is 0.00803
Epoch85, test_loss is 1.96290, train_accuracy is 0.99975,test_accuracy is 0.48625,train_all_loss is 0.36020,source_cla_loss is 0.35000,domain_loss is 0.00556
Epoch86, test_loss is 1.99381, train_accuracy is 1.00000,test_accuracy is 0.48825,train_all_loss is 0.35978,source_cla_loss is 0.34941,domain_loss is 0.00562
Epoch87, test_loss is 1.92070, train_accuracy is 1.00000,test_accuracy is 0.47525,train_all_loss is 0.36388,source_cla_loss is 0.34945,domain_loss is 0.00778
Epoch88, test_loss is 1.90456, train_accuracy is 1.00000,test_accuracy is 0.49850,train_all_loss is 0.35838,source_cla_loss is 0.34956,domain_loss is 0.00473
Epoch89, test_loss is 1.94458, train_accuracy is 0.99975,test_accuracy is 0.47600,train_all_loss is 0.36450,source_cla_loss is 0.35095,domain_loss is 0.00723
Epoch90, test_loss is 2.07750, train_accuracy is 1.00000,test_accuracy is 0.44650,train_all_loss is 0.36025,source_cla_loss is 0.34988,domain_loss is 0.00550
Epoch91, test_loss is 2.09842, train_accuracy is 1.00000,test_accuracy is 0.42175,train_all_loss is 0.36093,source_cla_loss is 0.34968,domain_loss is 0.00594
Epoch92, test_loss is 2.12722, train_accuracy is 1.00000,test_accuracy is 0.40625,train_all_loss is 0.36386,source_cla_loss is 0.34979,domain_loss is 0.00738
Epoch93, test_loss is 1.78110, train_accuracy is 0.99950,test_accuracy is 0.51150,train_all_loss is 0.36337,source_cla_loss is 0.35180,domain_loss is 0.00604
Epoch94, test_loss is 1.88155, train_accuracy is 0.99950,test_accuracy is 0.50150,train_all_loss is 0.36641,source_cla_loss is 0.35073,domain_loss is 0.00815
Epoch95, test_loss is 1.89918, train_accuracy is 1.00000,test_accuracy is 0.47300,train_all_loss is 0.36335,source_cla_loss is 0.35031,domain_loss is 0.00674
Epoch96, test_loss is 1.87588, train_accuracy is 1.00000,test_accuracy is 0.48575,train_all_loss is 0.36017,source_cla_loss is 0.34973,domain_loss is 0.00537
Epoch97, test_loss is 1.81331, train_accuracy is 1.00000,test_accuracy is 0.50875,train_all_loss is 0.35987,source_cla_loss is 0.34948,domain_loss is 0.00531
Epoch98, test_loss is 1.73135, train_accuracy is 1.00000,test_accuracy is 0.54350,train_all_loss is 0.36217,source_cla_loss is 0.34973,domain_loss is 0.00633
Epoch99, test_loss is 1.90660, train_accuracy is 0.99975,test_accuracy is 0.45425,train_all_loss is 0.36547,source_cla_loss is 0.35049,domain_loss is 0.00758
Epoch100, test_loss is 2.04686, train_accuracy is 1.00000,test_accuracy is 0.43025,train_all_loss is 0.36341,source_cla_loss is 0.35029,domain_loss is 0.00661
Epoch101, test_loss is 2.08754, train_accuracy is 1.00000,test_accuracy is 0.43425,train_all_loss is 0.35962,source_cla_loss is 0.34967,domain_loss is 0.00499
Epoch102, test_loss is 2.05368, train_accuracy is 1.00000,test_accuracy is 0.43450,train_all_loss is 0.36238,source_cla_loss is 0.34966,domain_loss is 0.00634
Epoch103, test_loss is 2.04378, train_accuracy is 1.00000,test_accuracy is 0.43300,train_all_loss is 0.36087,source_cla_loss is 0.34939,domain_loss is 0.00570
Epoch104, test_loss is 1.98537, train_accuracy is 1.00000,test_accuracy is 0.44825,train_all_loss is 0.36309,source_cla_loss is 0.34957,domain_loss is 0.00668
Epoch105, test_loss is 1.87151, train_accuracy is 1.00000,test_accuracy is 0.48725,train_all_loss is 0.36180,source_cla_loss is 0.34969,domain_loss is 0.00595
Epoch106, test_loss is 1.80578, train_accuracy is 1.00000,test_accuracy is 0.52875,train_all_loss is 0.36012,source_cla_loss is 0.34952,domain_loss is 0.00518
Epoch107, test_loss is 1.88614, train_accuracy is 1.00000,test_accuracy is 0.49150,train_all_loss is 0.35935,source_cla_loss is 0.34962,domain_loss is 0.00473
Epoch108, test_loss is 2.01156, train_accuracy is 1.00000,test_accuracy is 0.46025,train_all_loss is 0.35981,source_cla_loss is 0.34938,domain_loss is 0.00505
Epoch109, test_loss is 2.02546, train_accuracy is 1.00000,test_accuracy is 0.46275,train_all_loss is 0.36216,source_cla_loss is 0.34942,domain_loss is 0.00614
Epoch110, test_loss is 1.92141, train_accuracy is 1.00000,test_accuracy is 0.49175,train_all_loss is 0.35969,source_cla_loss is 0.34943,domain_loss is 0.00492
Epoch111, test_loss is 1.62420, train_accuracy is 0.99975,test_accuracy is 0.55300,train_all_loss is 0.36185,source_cla_loss is 0.35010,domain_loss is 0.00561
Epoch112, test_loss is 1.82876, train_accuracy is 0.99975,test_accuracy is 0.51075,train_all_loss is 0.36234,source_cla_loss is 0.35144,domain_loss is 0.00517
Epoch113, test_loss is 1.92894, train_accuracy is 1.00000,test_accuracy is 0.46050,train_all_loss is 0.36150,source_cla_loss is 0.34964,domain_loss is 0.00561
Epoch114, test_loss is 1.93551, train_accuracy is 1.00000,test_accuracy is 0.48225,train_all_loss is 0.36063,source_cla_loss is 0.34962,domain_loss is 0.00518
Epoch115, test_loss is 1.78969, train_accuracy is 1.00000,test_accuracy is 0.52700,train_all_loss is 0.35976,source_cla_loss is 0.34965,domain_loss is 0.00473
Epoch116, test_loss is 1.35264, train_accuracy is 1.00000,test_accuracy is 0.65150,train_all_loss is 0.36248,source_cla_loss is 0.34978,domain_loss is 0.00592
Epoch117, test_loss is 1.30626, train_accuracy is 0.99950,test_accuracy is 0.59500,train_all_loss is 0.36203,source_cla_loss is 0.35149,domain_loss is 0.00489
Epoch118, test_loss is 1.06381, train_accuracy is 0.99975,test_accuracy is 0.70575,train_all_loss is 0.35764,source_cla_loss is 0.35069,domain_loss is 0.00321
Epoch119, test_loss is 0.72888, train_accuracy is 1.00000,test_accuracy is 0.84975,train_all_loss is 0.35527,source_cla_loss is 0.34929,domain_loss is 0.00275
Epoch120, test_loss is 1.52062, train_accuracy is 1.00000,test_accuracy is 0.57600,train_all_loss is 0.35413,source_cla_loss is 0.34946,domain_loss is 0.00214
Epoch121, test_loss is 1.29272, train_accuracy is 1.00000,test_accuracy is 0.61100,train_all_loss is 0.35314,source_cla_loss is 0.34946,domain_loss is 0.00168
Epoch122, test_loss is 0.77498, train_accuracy is 1.00000,test_accuracy is 0.79200,train_all_loss is 0.35075,source_cla_loss is 0.34946,domain_loss is 0.00058
Epoch123, test_loss is 0.35446, train_accuracy is 1.00000,test_accuracy is 0.99875,train_all_loss is 0.34952,source_cla_loss is 0.34924,domain_loss is 0.00012
Epoch124, test_loss is 0.36260, train_accuracy is 1.00000,test_accuracy is 0.99500,train_all_loss is 0.34924,source_cla_loss is 0.34909,domain_loss is 0.00007
Epoch125, test_loss is 0.35317, train_accuracy is 1.00000,test_accuracy is 0.99925,train_all_loss is 0.34930,source_cla_loss is 0.34913,domain_loss is 0.00007
Epoch126, test_loss is 0.35170, train_accuracy is 1.00000,test_accuracy is 0.99950,train_all_loss is 0.34928,source_cla_loss is 0.34907,domain_loss is 0.00010
Epoch127, test_loss is 0.35829, train_accuracy is 1.00000,test_accuracy is 0.99700,train_all_loss is 0.34932,source_cla_loss is 0.34914,domain_loss is 0.00008
Epoch128, test_loss is 0.35118, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34908,source_cla_loss is 0.34897,domain_loss is 0.00005
Epoch129, test_loss is 0.35110, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34901,source_cla_loss is 0.34893,domain_loss is 0.00003
Epoch130, test_loss is 0.35183, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34904,source_cla_loss is 0.34894,domain_loss is 0.00004
Epoch131, test_loss is 0.35454, train_accuracy is 1.00000,test_accuracy is 0.99900,train_all_loss is 0.34905,source_cla_loss is 0.34896,domain_loss is 0.00004
Epoch132, test_loss is 0.45340, train_accuracy is 1.00000,test_accuracy is 0.94450,train_all_loss is 0.34921,source_cla_loss is 0.34904,domain_loss is 0.00008
Epoch133, test_loss is 0.34977, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34935,source_cla_loss is 0.34913,domain_loss is 0.00010
Epoch134, test_loss is 0.35017, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34920,source_cla_loss is 0.34906,domain_loss is 0.00006
Epoch135, test_loss is 0.35451, train_accuracy is 1.00000,test_accuracy is 0.99900,train_all_loss is 0.34912,source_cla_loss is 0.34899,domain_loss is 0.00005
Epoch136, test_loss is 0.34991, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34903,source_cla_loss is 0.34893,domain_loss is 0.00004
Epoch137, test_loss is 0.34986, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34906,source_cla_loss is 0.34896,domain_loss is 0.00004
Epoch138, test_loss is 0.35165, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34928,source_cla_loss is 0.34916,domain_loss is 0.00005
Epoch139, test_loss is 0.35202, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34909,source_cla_loss is 0.34899,domain_loss is 0.00004
Epoch140, test_loss is 0.35071, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34912,source_cla_loss is 0.34902,domain_loss is 0.00004
Epoch141, test_loss is 0.35684, train_accuracy is 1.00000,test_accuracy is 0.99900,train_all_loss is 0.34903,source_cla_loss is 0.34897,domain_loss is 0.00003
Epoch142, test_loss is 0.35529, train_accuracy is 1.00000,test_accuracy is 0.99875,train_all_loss is 0.34911,source_cla_loss is 0.34898,domain_loss is 0.00006
Epoch143, test_loss is 0.35079, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34928,source_cla_loss is 0.34908,domain_loss is 0.00008
Epoch144, test_loss is 0.35015, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34987,source_cla_loss is 0.34964,domain_loss is 0.00009
Epoch145, test_loss is 0.37696, train_accuracy is 1.00000,test_accuracy is 0.98600,train_all_loss is 0.34924,source_cla_loss is 0.34913,domain_loss is 0.00005
Epoch146, test_loss is 0.35011, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34918,source_cla_loss is 0.34900,domain_loss is 0.00007
Epoch147, test_loss is 0.35067, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34922,source_cla_loss is 0.34910,domain_loss is 0.00005
Epoch148, test_loss is 0.34959, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34979,source_cla_loss is 0.34955,domain_loss is 0.00010
Epoch149, test_loss is 0.35497, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34996,source_cla_loss is 0.34975,domain_loss is 0.00008
Epoch150, test_loss is 0.35252, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34923,source_cla_loss is 0.34914,domain_loss is 0.00004
Epoch151, test_loss is 0.35072, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34911,source_cla_loss is 0.34906,domain_loss is 0.00002
Epoch152, test_loss is 0.35047, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34921,source_cla_loss is 0.34909,domain_loss is 0.00005
Epoch153, test_loss is 0.35013, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34905,source_cla_loss is 0.34900,domain_loss is 0.00002
Epoch154, test_loss is 0.34926, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34891,domain_loss is 0.00002
Epoch155, test_loss is 0.34921, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34893,domain_loss is 0.00002
Epoch156, test_loss is 0.34936, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34896,source_cla_loss is 0.34890,domain_loss is 0.00002
Epoch157, test_loss is 0.34949, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34897,source_cla_loss is 0.34892,domain_loss is 0.00002
Epoch158, test_loss is 0.34983, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34905,source_cla_loss is 0.34899,domain_loss is 0.00002
Epoch159, test_loss is 0.34976, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34891,source_cla_loss is 0.34887,domain_loss is 0.00002
Epoch160, test_loss is 0.34934, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34889,domain_loss is 0.00003
Epoch161, test_loss is 0.34963, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34897,source_cla_loss is 0.34890,domain_loss is 0.00003
Epoch162, test_loss is 0.35000, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34905,source_cla_loss is 0.34897,domain_loss is 0.00003
Epoch163, test_loss is 0.34932, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34903,source_cla_loss is 0.34898,domain_loss is 0.00002
Epoch164, test_loss is 0.34963, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34893,source_cla_loss is 0.34887,domain_loss is 0.00002
Epoch165, test_loss is 0.34932, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34903,source_cla_loss is 0.34895,domain_loss is 0.00003
Epoch166, test_loss is 0.34944, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34891,domain_loss is 0.00002
Epoch167, test_loss is 0.35002, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34903,source_cla_loss is 0.34896,domain_loss is 0.00002
Epoch168, test_loss is 0.34956, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34908,source_cla_loss is 0.34901,domain_loss is 0.00003
Epoch169, test_loss is 0.34920, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34905,source_cla_loss is 0.34899,domain_loss is 0.00002
Epoch170, test_loss is 0.34940, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34893,source_cla_loss is 0.34888,domain_loss is 0.00002
Epoch171, test_loss is 0.34942, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34893,source_cla_loss is 0.34889,domain_loss is 0.00001
Epoch172, test_loss is 0.35001, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34895,source_cla_loss is 0.34889,domain_loss is 0.00002
Epoch173, test_loss is 0.35019, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34917,source_cla_loss is 0.34906,domain_loss is 0.00004
Epoch174, test_loss is 0.35418, train_accuracy is 1.00000,test_accuracy is 0.99950,train_all_loss is 0.34925,source_cla_loss is 0.34917,domain_loss is 0.00003
Epoch175, test_loss is 0.34981, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34906,source_cla_loss is 0.34902,domain_loss is 0.00002
Epoch176, test_loss is 0.34910, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34894,domain_loss is 0.00002
Epoch177, test_loss is 0.35157, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34903,source_cla_loss is 0.34895,domain_loss is 0.00003
Epoch178, test_loss is 0.36411, train_accuracy is 1.00000,test_accuracy is 0.99900,train_all_loss is 0.34918,source_cla_loss is 0.34911,domain_loss is 0.00002
Epoch179, test_loss is 0.35386, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.35033,source_cla_loss is 0.34998,domain_loss is 0.00012
Epoch180, test_loss is 0.39304, train_accuracy is 1.00000,test_accuracy is 0.97850,train_all_loss is 0.34949,source_cla_loss is 0.34938,domain_loss is 0.00004
Epoch181, test_loss is 0.35160, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34927,source_cla_loss is 0.34917,domain_loss is 0.00003
Epoch182, test_loss is 0.34962, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34908,source_cla_loss is 0.34901,domain_loss is 0.00002
Epoch183, test_loss is 0.34926, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34891,domain_loss is 0.00002
Epoch184, test_loss is 0.49573, train_accuracy is 1.00000,test_accuracy is 0.92900,train_all_loss is 0.34966,source_cla_loss is 0.34932,domain_loss is 0.00011
Epoch185, test_loss is 0.35219, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.35185,source_cla_loss is 0.35073,domain_loss is 0.00036
Epoch186, test_loss is 0.35016, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34945,source_cla_loss is 0.34929,domain_loss is 0.00005
Epoch187, test_loss is 0.34990, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34914,source_cla_loss is 0.34902,domain_loss is 0.00004
Epoch188, test_loss is 0.35030, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34904,source_cla_loss is 0.34896,domain_loss is 0.00003
Epoch189, test_loss is 0.35148, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34905,source_cla_loss is 0.34897,domain_loss is 0.00003
Epoch190, test_loss is 0.35053, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34895,source_cla_loss is 0.34891,domain_loss is 0.00001
Epoch191, test_loss is 0.34955, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34896,source_cla_loss is 0.34888,domain_loss is 0.00002
Epoch192, test_loss is 0.34971, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34894,domain_loss is 0.00001
Epoch193, test_loss is 0.34957, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34905,source_cla_loss is 0.34896,domain_loss is 0.00003
Epoch194, test_loss is 0.34920, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34892,domain_loss is 0.00002
Epoch195, test_loss is 0.35196, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34898,source_cla_loss is 0.34891,domain_loss is 0.00002
Epoch196, test_loss is 0.35668, train_accuracy is 1.00000,test_accuracy is 0.99975,train_all_loss is 0.34921,source_cla_loss is 0.34903,domain_loss is 0.00005
Epoch197, test_loss is 0.35010, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34966,source_cla_loss is 0.34950,domain_loss is 0.00005
Epoch198, test_loss is 0.34923, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34955,source_cla_loss is 0.34939,domain_loss is 0.00004
Epoch199, test_loss is 0.35031, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34922,source_cla_loss is 0.34910,domain_loss is 0.00003
Epoch200, test_loss is 0.35095, train_accuracy is 1.00000,test_accuracy is 1.00000,train_all_loss is 0.34909,source_cla_loss is 0.34898,domain_loss is 0.00003
```

# Contact

- **Chao He**
- **chaohe#bjtu.edu.cn   (please replace # by @)**
