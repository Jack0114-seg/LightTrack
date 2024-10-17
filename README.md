lighttrack 目前环境可以运行
丢失框排查：
    1：模型尺度排序（未开始）
    2：后处理（处理中）

问题：
  未对齐onnx，这段时间会继续搞一下这个方法。


---------START QUERY MODEL ATTRBUTE---------

net-init model input num: 1, output num: 1
init model attrbute:
  index=0, name=input1, n_dims=4, dims=[1, 127, 127, 3], n_elems=48387, size=96774, fmt=NHWC, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

  index=0, name=output.1, n_dims=4, dims=[1, 96, 8, 8], n_elems=6144, size=12288, fmt=NCHW, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

backbone model input num: 1, output num: 1
  index=0, name=input1, n_dims=4, dims=[1, 288, 288, 3], n_elems=248832, size=497664, fmt=NHWC, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

  index=0, name=output.1, n_dims=4, dims=[1, 96, 18, 18], n_elems=31104, size=62208, fmt=NCHW, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

neck_head model input num: 2, output num: 2
  index=0, name=input1, n_dims=4, dims=[1, 8, 8, 96], n_elems=6144, size=12288, fmt=NHWC, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

  index=0, name=output.1, n_dims=4, dims=[1, 1, 18, 18], n_elems=324, size=648, fmt=NCHW, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

  index=1, name=input2, n_dims=4, dims=[1, 18, 18, 96], n_elems=31104, size=62208, fmt=NHWC, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

  index=1, name=output.2, n_dims=4, dims=[1, 4, 18, 18], n_elems=1296, size=2592, fmt=NCHW, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000

---------START QUERY SDK/Driver VERSION---------

sdk api version: 1.6.0 (9a7b5d24c@2023-12-13T17:31:11)
driver version: 0.8.0
