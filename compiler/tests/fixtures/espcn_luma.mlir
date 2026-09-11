"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x4x6x4xi8>) -> tensor<1x8x12x1xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x4x6x4xi8>):
    %0 = "tosa.const"() <{values = dense<[[[[3, -3, 3, 2], [0, 1, -1, 4], [2, 4, -4, -2]], [[-1, -4, -4, 3], [2, 0, -2, -2], [3, -1, -1, -3]], [[3, 4, -2, 1], [-4, 0, 1, 0], [1, 2, 4, -3]]], [[[-3, 0, -4, 4], [0, 0, 2, 2], [2, -2, 3, 3]], [[-4, 2, 2, 4], [-2, -2, 1, 3], [0, -1, 4, 1]], [[3, 1, -3, -4], [3, -4, 2, 2], [2, 0, -2, -3]]], [[[1, -3, -4, -4], [-2, -2, -1, 4], [3, -2, -4, 1]], [[2, -3, -2, 1], [4, 1, 4, 2], [4, 3, -4, 1]], [[1, 1, 3, 2], [-2, 3, -2, 1], [3, 2, 1, -2]]], [[[-3, -2, -2, -4], [3, 4, -4, 0], [-4, 0, 1, -3]], [[3, -2, -4, 3], [2, 0, -2, -2], [-1, 3, -3, 4]], [[4, -4, -2, 1], [-3, -3, -2, -2], [-2, -1, -4, -2]]]]> : tensor<4x3x3x4xi8>}> : () -> tensor<4x3x3x4xi8>
    %1 = "tosa.const"() <{values = dense<[19, -9, -7, 5]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.conv2d"(%arg0, %0, %1, %2, %3) <{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x4x6x4xi8>, tensor<4x3x3x4xi8>, tensor<4xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x6x4xi32>
    %5 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<34> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = "tosa.rescale"(%4, %5, %6, %7, %8) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x4x6x4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x4x6x4xi8>
    %10 = "tosa.clamp"(%9) <{max_val = 127 : i8, min_val = 0 : i8, nan_mode = #tosa.nan_mode<PROPAGATE>}> : (tensor<1x4x6x4xi8>) -> tensor<1x4x6x4xi8>
    %ds0 = "tosa.const_shape"() <{values = dense<[1, 4, 6, 1, 2, 2]> : tensor<6xindex>}> : () -> !tosa.shape<6>
    %ds1 = "tosa.const_shape"() <{values = dense<[1, 8, 12, 1]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %d0 = "tosa.reshape"(%10, %ds0) : (tensor<1x4x6x4xi8>, !tosa.shape<6>) -> tensor<1x4x6x1x2x2xi8>
    %d1 = "tosa.transpose"(%d0) <{perms = array<i32: 0, 1, 4, 2, 5, 3>}> : (tensor<1x4x6x1x2x2xi8>) -> tensor<1x4x2x6x2x1xi8>
    %dout = "tosa.reshape"(%d1, %ds1) : (tensor<1x4x2x6x2x1xi8>, !tosa.shape<4>) -> tensor<1x8x12x1xi8>
    "func.return"(%dout) : (tensor<1x8x12x1xi8>) -> ()
  }) : () -> ()
}) : () -> ()
