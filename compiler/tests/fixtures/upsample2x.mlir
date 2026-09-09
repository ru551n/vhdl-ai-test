"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x3x5x8xi8>) -> tensor<1x6x10x8xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x3x5x8xi8>):
    %c0_s0 = "tosa.const_shape"() <{values = dense<[1, 3, 1, 5, 8]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s1 = "tosa.const_shape"() <{values = dense<[1, 1, 2, 1, 1]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s2 = "tosa.const_shape"() <{values = dense<[1, 6, 5, 1, 8]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s3 = "tosa.const_shape"() <{values = dense<[1, 1, 1, 2, 1]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s4 = "tosa.const_shape"() <{values = dense<[1, 6, 10, 8]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %c0_0 = "tosa.reshape"(%arg0, %c0_s0) : (tensor<1x3x5x8xi8>, !tosa.shape<5>) -> tensor<1x3x1x5x8xi8>
    %c0_1 = "tosa.tile"(%c0_0, %c0_s1) : (tensor<1x3x1x5x8xi8>, !tosa.shape<5>) -> tensor<1x3x2x5x8xi8>
    %c0_2 = "tosa.reshape"(%c0_1, %c0_s2) : (tensor<1x3x2x5x8xi8>, !tosa.shape<5>) -> tensor<1x6x5x1x8xi8>
    %c0_3 = "tosa.tile"(%c0_2, %c0_s3) : (tensor<1x6x5x1x8xi8>, !tosa.shape<5>) -> tensor<1x6x5x2x8xi8>
    %u0 = "tosa.reshape"(%c0_3, %c0_s4) : (tensor<1x6x5x2x8xi8>, !tosa.shape<4>) -> tensor<1x6x10x8xi8>
    "func.return"(%u0) : (tensor<1x6x10x8xi8>) -> ()
  }) : () -> ()
}) : () -> ()
