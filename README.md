# PCCLS

## Models

### PointNetV0 Series

#### PointNetV00

添加 PointNetV00 并将其改为 4 阶段的模型 (删除其中 STN3d 部分的内容), 虽然会降点, 但是便于作为后续模型的 BaseLine.

#### PointNetV01

使用四层残差连接替换 PointNetV00 中的全连接.

#### PointNetV02

对 PointNetV00 添加 PointClsStem (邻域特征聚合) 用于增强初始的特征.

## Experiments

所有模型结果均为训练 10 次的平均值

### ModelNet40

|   Model    |  Avg Best OA (\%)  | Max Best OA (\%) | Avg Best mAcc (\%) | Max Best mAcc (\%) |
| :--------: | :----------------: | :--------------: | :----------------: | :----------------: |


### ScanObjectNN

|   Method   |  Avg Best OA (\%)  | Max Best OA (\%) | Avg Best mAcc (\%) | Max Best mAcc (\%) |
| :--------: | :----------------: | :--------------: | :----------------: | :----------------: |