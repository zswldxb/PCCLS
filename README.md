# PCCLS

## Models

### PointNetV1

添加 PointNetV1 并将其改为 4 阶段的模型 (删除其中 STN3d 部分的内容), 虽然会降点, 但是便于作为后续模型的 BaseLine.

### PointNetV2

使用四层残差连接替换 PointNetV1 中的全连接.

## Experiments

所有模型结果均为训练 10 次的平均值

### ModelNet40

|   Model    |  Avg Best OA (\%)  | Max Best OA (\%) | Avg Best mAcc (\%) | Max Best mAcc (\%) |
| :--------: | :----------------: | :--------------: | :----------------: | :----------------: |
| PointNetV1 | $88.691 \pm 0.234$ |     $89.182$     | $83.211 \pm 0.280$ |      $83.636$      |
| PointNetV2 | $88.780 \pm 0.274$ |     $89.182$     | $83.578 \pm 0.282$ |      $84.123$      |


### ScanObjectNN

|   Method   |  Avg Best OA (\%)  | Max Best OA (\%) | Avg Best mAcc (\%) | Max Best mAcc (\%) |
| :--------: | :----------------: | :--------------: | :----------------: | :----------------: |
| PointNetV1 | $66.242 \pm 0.512$ |     $67.349$     | $57.999 \pm 0.364$ |      $58.970$      |