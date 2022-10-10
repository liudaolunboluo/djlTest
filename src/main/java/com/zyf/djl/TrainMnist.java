package com.zyf.djl;

import ai.djl.Model;
import ai.djl.basicdataset.cv.ImageDataset;
import ai.djl.basicdataset.cv.classification.ImageClassificationDataset;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * @author zhangyunfan@fiture.com
 * @version 1.0
 * @ClassName: TrainMnist
 * @date 2022/10/9
 */
public final class TrainMnist {

    public static void main(String[] args) throws IOException, TranslateException {
        int batchSize = 32;
        //数据集，自带数据集
        Mnist mnist = Mnist.builder().setSampling(batchSize, true).build();
        mnist.prepare(new ProgressBar());
        try (Model model = Model.newInstance("mlp")) {
            //输入层，输出层，隐藏层
            model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));
            //训练配置
            //softmaxCrossEntropyLoss 是分类问题的标准损失
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    // 使用准确性，以便我们人类可以了解模型的准确性
                    .addEvaluator(new Accuracy()).addTrainingListeners(TrainingListener.Defaults.logging());
            // 现在我们有了训练配置，我们应该为我们的模型创建一个新的训练器
            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(1, 28 * 28));
            //深度学习通常在 epoch 中进行训练，每个 epoch 在数据集中的每个项目上训练模型一次。
            int epoch = 2;
            //开始训练,传入训练师、epocj、数据集就可以了
            EasyTrain.fit(trainer, epoch, mnist, null);
            //保存模型
            Path modelDir = Paths.get("build/mlp");
            Files.createDirectories(modelDir);
            model.setProperty("Epoch", String.valueOf(epoch));
            model.save(modelDir, "mlp");
        }
    }

}
