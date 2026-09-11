"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x3x5x32xi8>) -> tensor<1x6x10x8xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x3x5x32xi8>):
    %ds0 = "tosa.const_shape"() <{values = dense<[1, 3, 5, 2, 2, 8]> : tensor<6xindex>}> : () -> !tosa.shape<6>
    %ds1 = "tosa.const_shape"() <{values = dense<[1, 6, 10, 8]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %d0 = "tosa.reshape"(%arg0, %ds0) : (tensor<1x3x5x32xi8>, !tosa.shape<6>) -> tensor<1x3x5x2x2x8xi8>
    %d1 = "tosa.transpose"(%d0) <{perms = array<i32: 0, 1, 3, 2, 4, 5>}> : (tensor<1x3x5x2x2x8xi8>) -> tensor<1x3x2x5x2x8xi8>
    %dout = "tosa.reshape"(%d1, %ds1) : (tensor<1x3x2x5x2x8xi8>, !tosa.shape<4>) -> tensor<1x6x10x8xi8>
    "func.return"(%dout) : (tensor<1x6x10x8xi8>) -> ()
  }) : () -> ()
}) : () -> ()
