"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x2x3x12xi8>) -> tensor<1x8x12x12xi8>, sym_name = "main"}> ({
  ^bb0(%arg0: tensor<1x2x3x12xi8>):
    %c0_s0 = "tosa.const_shape"() <{values = dense<[1, 2, 1, 3, 12]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s1 = "tosa.const_shape"() <{values = dense<[1, 1, 2, 1, 1]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s2 = "tosa.const_shape"() <{values = dense<[1, 4, 3, 1, 12]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s3 = "tosa.const_shape"() <{values = dense<[1, 1, 1, 2, 1]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c0_s4 = "tosa.const_shape"() <{values = dense<[1, 4, 6, 12]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %c0_0 = "tosa.reshape"(%arg0, %c0_s0) : (tensor<1x2x3x12xi8>, !tosa.shape<5>) -> tensor<1x2x1x3x12xi8>
    %c0_1 = "tosa.tile"(%c0_0, %c0_s1) : (tensor<1x2x1x3x12xi8>, !tosa.shape<5>) -> tensor<1x2x2x3x12xi8>
    %c0_2 = "tosa.reshape"(%c0_1, %c0_s2) : (tensor<1x2x2x3x12xi8>, !tosa.shape<5>) -> tensor<1x4x3x1x12xi8>
    %c0_3 = "tosa.tile"(%c0_2, %c0_s3) : (tensor<1x4x3x1x12xi8>, !tosa.shape<5>) -> tensor<1x4x3x2x12xi8>
    %u0 = "tosa.reshape"(%c0_3, %c0_s4) : (tensor<1x4x3x2x12xi8>, !tosa.shape<4>) -> tensor<1x4x6x12xi8>
    %c1_s0 = "tosa.const_shape"() <{values = dense<[1, 4, 1, 6, 12]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c1_s1 = "tosa.const_shape"() <{values = dense<[1, 1, 2, 1, 1]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c1_s2 = "tosa.const_shape"() <{values = dense<[1, 8, 6, 1, 12]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c1_s3 = "tosa.const_shape"() <{values = dense<[1, 1, 1, 2, 1]> : tensor<5xindex>}> : () -> !tosa.shape<5>
    %c1_s4 = "tosa.const_shape"() <{values = dense<[1, 8, 12, 12]> : tensor<4xindex>}> : () -> !tosa.shape<4>
    %c1_0 = "tosa.reshape"(%u0, %c1_s0) : (tensor<1x4x6x12xi8>, !tosa.shape<5>) -> tensor<1x4x1x6x12xi8>
    %c1_1 = "tosa.tile"(%c1_0, %c1_s1) : (tensor<1x4x1x6x12xi8>, !tosa.shape<5>) -> tensor<1x4x2x6x12xi8>
    %c1_2 = "tosa.reshape"(%c1_1, %c1_s2) : (tensor<1x4x2x6x12xi8>, !tosa.shape<5>) -> tensor<1x8x6x1x12xi8>
    %c1_3 = "tosa.tile"(%c1_2, %c1_s3) : (tensor<1x8x6x1x12xi8>, !tosa.shape<5>) -> tensor<1x8x6x2x12xi8>
    %u1 = "tosa.reshape"(%c1_3, %c1_s4) : (tensor<1x8x6x2x12xi8>, !tosa.shape<4>) -> tensor<1x8x12x12xi8>
    "func.return"(%u1) : (tensor<1x8x12x12xi8>) -> ()
  }) : () -> ()
}) : () -> ()
