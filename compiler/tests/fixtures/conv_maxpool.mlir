"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x8x8x4xi8>) -> tensor<1x4x4x8xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x8x8x4xi8>):
    %0 = "tosa.const"() <{values = dense<1> : tensor<8x3x3x4xi8>}> : () -> tensor<8x3x3x4xi8>
    %1 = "tosa.const"() <{values = dense<0> : tensor<8xi32>}> : () -> tensor<8xi32>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.conv2d"(%arg0, %0, %1, %2, %3) <{acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x4xi8>, tensor<8x3x3x4xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x8xi32>
    %5 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = "tosa.rescale"(%4, %5, %6, %7, %8) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x8x8x8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x8xi8>
    %10 = "tosa.clamp"(%9) <{max_val = 127 : i8, min_val = 0 : i8, nan_mode = #tosa.nan_mode<PROPAGATE>}> : (tensor<1x8x8x8xi8>) -> tensor<1x8x8x8xi8>
    %11 = "tosa.max_pool2d"(%10) <{kernel = array<i64: 2, 2>, nan_mode = #tosa.nan_mode<PROPAGATE>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}> : (tensor<1x8x8x8xi8>) -> tensor<1x4x4x8xi8>
    "func.return"(%11) : (tensor<1x4x4x8xi8>) -> ()
  }) : () -> ()
}) : () -> ()
