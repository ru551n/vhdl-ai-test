"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x8x8x8xi8>) -> tensor<1x8x8x8xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x8x8x8xi8>):
    %0 = "tosa.max_pool2d"(%arg0) <{kernel = array<i64: 5, 5>, nan_mode = #tosa.nan_mode<PROPAGATE>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>}> : (tensor<1x8x8x8xi8>) -> tensor<1x8x8x8xi8>
    "func.return"(%0) : (tensor<1x8x8x8xi8>) -> ()
  }) : () -> ()
}) : () -> ()
