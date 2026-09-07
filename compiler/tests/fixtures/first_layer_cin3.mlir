"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x8x8x3xi8>) -> tensor<1x4x4x8xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x8x8x3xi8>):
    %0 = "tosa.const"() <{values = dense<[[[[-2, -1, -3], [-3, -1, -1], [-1, 4, 2]], [[3, 0, 2], [0, -4, 3], [-4, 4, -4]], [[0, 1, 4], [-2, -2, 0], [0, 4, 4]]], [[[1, -1, 2], [1, -3, 4], [4, -2, -1]], [[2, -3, 0], [-3, -3, -1], [-3, -3, -4]], [[0, 1, 2], [-4, 2, 1], [-2, 4, -4]]], [[[-1, 1, 4], [1, -3, 2], [-2, 0, 1]], [[4, 1, -1], [3, -2, 1], [3, 2, -3]], [[-2, -1, 3], [-3, 0, 4], [2, 3, 3]]], [[[3, 1, -4], [3, -2, 4], [-1, -4, 3]], [[4, -2, -2], [-2, 4, -3], [1, 1, -1]], [[3, -4, 0], [-2, -4, 0], [3, 3, 2]]], [[[-2, 1, -2], [1, 2, 4], [-2, 1, -1]], [[2, -1, 3], [2, -3, 2], [-1, 2, 4]], [[4, -3, 3], [-1, -1, 1], [1, 3, -3]]], [[[-3, 0, -1], [4, 4, 0], [-1, -3, 1]], [[-1, -1, 1], [-2, 3, 4], [3, -4, 0]], [[-4, -3, 0], [0, -2, 0], [-3, 1, -4]]], [[[-3, 3, 1], [3, 0, -1], [0, 0, -2]], [[-2, 0, 2], [2, 0, -3], [-1, -2, -1]], [[-3, 3, 3], [-3, 0, -3], [1, 3, -4]]], [[[4, -4, 1], [-2, 1, 1], [-1, 2, 1]], [[4, -2, -3], [0, -3, -4], [-4, 1, -2]], [[-4, 2, 1], [-4, 4, 4], [1, 1, -1]]]]> : tensor<8x3x3x3xi8>}> : () -> tensor<8x3x3x3xi8>
    %1 = "tosa.const"() <{values = dense<[12, 11, 15, 8, -12, 20, 5, 2]> : tensor<8xi32>}> : () -> tensor<8xi32>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.conv2d"(%arg0, %0, %1, %2, %3) <{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>}> : (tensor<1x8x8x3xi8>, tensor<8x3x3x3xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x8xi32>
    %5 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<34> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = "tosa.rescale"(%4, %5, %6, %7, %8) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x4x4x8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x4x4x8xi8>
    %10 = "tosa.clamp"(%9) <{max_val = 127 : i8, min_val = 0 : i8, nan_mode = #tosa.nan_mode<PROPAGATE>}> : (tensor<1x4x4x8xi8>) -> tensor<1x4x4x8xi8>
    "func.return"(%10) : (tensor<1x4x4x8xi8>) -> ()
  }) : () -> ()
}) : () -> ()
