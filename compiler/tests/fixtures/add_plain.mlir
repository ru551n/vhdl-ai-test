"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x8x8x8xi8>, tensor<1x8x8x8xi8>) -> tensor<1x8x8x8xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x8x8x8xi8>, %arg1: tensor<1x8x8x8xi8>):
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x8x8x8xi8>, tensor<1x8x8x8xi8>) -> tensor<1x8x8x8xi8>
    "func.return"(%0) : (tensor<1x8x8x8xi8>) -> ()
  }) : () -> ()
}) : () -> ()
