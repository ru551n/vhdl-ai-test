"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x4x6x12xi8>, tensor<1x4x6x12xi8>) -> tensor<1x4x6x12xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x4x6x12xi8>, %arg1: tensor<1x4x6x12xi8>):
    %0 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.rescale"(%arg0, %0, %1, %2, %3) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x4x6x12xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x6x12xi8>
    %5 = "tosa.rescale"(%arg1, %0, %1, %2, %3) <{input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = #tosa.rounding_mode<SINGLE_ROUND>, scale32 = true}> : (tensor<1x4x6x12xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x6x12xi8>
    %6 = "tosa.add"(%4, %5) : (tensor<1x4x6x12xi8>, tensor<1x4x6x12xi8>) -> tensor<1x4x6x12xi8>
    "func.return"(%6) : (tensor<1x4x6x12xi8>) -> ()
  }) : () -> ()
}) : () -> ()
