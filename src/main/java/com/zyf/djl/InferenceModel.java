package com.zyf.djl;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author zhangyunfan@fiture.com
 * @version 1.0
 * @ClassName: InferenceModel
 * @Description: TODO
 * @date 2022/10/9
 */
public class InferenceModel {

    public static void main(String[] args) throws Exception {
        Image oldImage = ImageFactory.getInstance().fromFile(Paths.get("/Users/zhangyunfan/Desktop/three.png"));
        NDArray resize = NDImageUtils.resize(oldImage.toNDArray(NDManager.newBaseManager()), 28, 28);
        Image image = ImageFactory.getInstance().fromNDArray(resize);
        image.getWrappedImage();
        Path modelDir = Paths.get("build/mlp");
        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));
            model.load(modelDir);
            Translator<Image, Classifications> translator = new Translator<Image, Classifications>() {

                @Override
                public NDList processInput(TranslatorContext ctx, Image input) {
                    // 将图像转换为 NDArray
                    NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);
                    return new NDList(NDImageUtils.toTensor(array));
                }

                @Override
                public Classifications processOutput(TranslatorContext ctx, NDList list) {
                    // 使用输出概率创建分类
                    NDArray probabilities = list.singletonOrThrow().softmax(0);
                    List<String> classNames = IntStream.range(0, 10).mapToObj(String::valueOf).collect(Collectors.toList());
                    return new Classifications(classNames, probabilities);
                }

                @Override
                public Batchifier getBatchifier() {
                    // 批处理器描述了如何将批处理组合在一起
                    // Stacking，最常见的批处理器，将 N [X1, X2, ...] 数组转换为单个 [N, X1, X2, ...] 数组
                    return Batchifier.STACK;
                }
            };
            Predictor<Image, Classifications> imageClassificationsPredictor = model.newPredictor(translator);
            Classifications classifications = imageClassificationsPredictor.predict(image);
            System.out.println(classifications);
            System.out.println("图片上的数字是："+classifications.best().getClassName()+"概率是:"+classifications.best().getProbability());
        }

    }
}
