package com.zyf.djl;

import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;

import java.util.function.Function;

/**
 * @author zhangyunfan@fiture.com
 * @version 1.0
 * @ClassName: Mlp
 * @Description: TODO
 * @date 2022/10/9
 */
public class Mlp extends SequentialBlock {

    /**
     * 使用 RELU 创建 MLP 神经网络。
     *
     * @param input  : 输入向量的大小
     * @param output :输出向量的大小
     * @param hidden :所有隐藏层的大小
     */
    public Mlp(int input, int output, int[] hidden) {
        this(input, output, hidden, Activation::relu);
    }

    /**
     * 创建 MLP 神经网络。
     *
     * @param input      :    输入向量的大小
     * @param output     :   输出向量的大小
     * @param hidden     :  所有隐藏层的大小
     * @param activation : 要使用的激活函数
     */
    public Mlp(int input, int output, int[] hidden, Function<NDList, NDList> activation) {
        add(Blocks.batchFlattenBlock(input));
        for (int hiddenSize : hidden) {
            add(Linear.builder().setUnits(hiddenSize).build());
            add(activation);
        }

        add(Linear.builder().setUnits(output).build());
    }
}