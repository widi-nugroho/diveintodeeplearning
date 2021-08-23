package cnn

import ai.djl.Device
import ai.djl.Model
import ai.djl.basicdataset.cv.classification.FashionMnist
import ai.djl.engine.Engine
import ai.djl.metric.Metric
import ai.djl.metric.Metrics
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.convolutional.Conv2d
import ai.djl.nn.core.Linear
import ai.djl.nn.pooling.Pool
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.Trainer
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.dataset.Dataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.evaluator.Evaluator
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.translate.TranslateException
import org.math.plot.Plot2DPanel
import java.awt.Color
import java.io.IOException
import javax.swing.JFrame


@Throws(IOException::class, TranslateException::class)
fun trainingChapter6(
    trainIter: ArrayDataset, testIter: ArrayDataset?,
    numEpochs: Int, trainer: Trainer
): List<DoubleArray> {
    var avgTrainTimePerEpoch = 0.0
    val evaluatorMetrics: MutableMap<String, DoubleArray> = HashMap()
    trainer.metrics = Metrics()
    EasyTrain.fit(trainer, numEpochs, trainIter, testIter)
    val metrics = trainer.metrics
    trainer.evaluators.stream()
        .forEach { evaluator: Evaluator ->
            evaluatorMetrics["train_epoch_" + evaluator.name] =
                metrics.getMetric("train_epoch_" + evaluator.name).stream()
                    .mapToDouble { x: Metric -> x.value.toDouble() }.toArray()
            evaluatorMetrics["validate_epoch_" + evaluator.name] =
                metrics.getMetric("validate_epoch_" + evaluator.name).stream()
                    .mapToDouble { x: Metric -> x.value.toDouble() }.toArray()
        }
    avgTrainTimePerEpoch = metrics.mean("epoch")
    var trainLoss = evaluatorMetrics["train_epoch_SoftmaxCrossEntropyLoss"]!!
    var trainAccuracy = evaluatorMetrics["train_epoch_Accuracy"]!!
    var testAccuracy = evaluatorMetrics["validate_epoch_Accuracy"]!!
    System.out.printf("loss %.3f,", trainLoss.get(numEpochs - 1))
    System.out.printf(" train acc %.3f,", trainAccuracy.get(numEpochs - 1))
    System.out.printf(" test acc %.3f\n", testAccuracy.get(numEpochs - 1))
    System.out.printf("%.1f examples/sec \n", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10.0, 9.0)))
    return listOf(trainLoss,trainAccuracy,testAccuracy)
}
fun trainingChapter62(trainIter:ArrayDataset,testIter:ArrayDataset,numEpochs:Int,trainer:Trainer): List<DoubleArray?> {
    var avgTrainTimePerEpoch:Double = 0.0
    val evaluatorMetrics: Map<String, DoubleArray> = HashMap()

    trainer.metrics = Metrics()

    EasyTrain.fit(trainer,numEpochs,trainIter,testIter)
    var metrics = trainer.metrics


    trainer.evaluators.stream()
        .forEach { evaluator ->
            evaluatorMetrics.plus(Pair("train_epoch_" + evaluator.name,metrics.getMetric("train_epoch_" + evaluator.name).stream()
                .mapToDouble { x: Metric -> x.value.toDouble() }.toArray()))
            evaluatorMetrics.plus(Pair("validate_epoch_" + evaluator.name,metrics.getMetric("validate_epoch_" + evaluator.name).stream()
                .mapToDouble { x: Metric -> x.value.toDouble() }.toArray()))
        }

    avgTrainTimePerEpoch = metrics.mean("epoch");

    var trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
    var trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
    var testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

    return listOf(trainLoss,trainAccuracy,testAccuracy)
}


fun main() {
    Engine.getInstance().setRandomSeed(1111)
    val manager = NDManager.newBaseManager()
    var block = SequentialBlock()
    block
        .add(Conv2d.builder()
            .setKernelShape(Shape(5,5))
            .optPadding(Shape(2,2))
            .optBias(false)
            .setFilters(6)
            .build())
        .add(Activation.sigmoidBlock())
        .add(Pool.avgPool2dBlock(Shape(5,5),Shape(2,2),Shape(2,2)))
        .add(Conv2d.builder()
            .setKernelShape(Shape(5,5))
            .setFilters(16)
            .build())
        .add(Activation.sigmoidBlock())
        .add(Pool.avgPool2dBlock(Shape(5,5),Shape(2,2),Shape(2,2)))

        .add(Blocks.batchFlattenBlock())
        .add(Linear.builder()
            .setUnits(120)
            .build())
        .add(Activation.sigmoidBlock())
        .add(Linear.builder()
            .setUnits(84)
            .build())
        .add(Activation.sigmoidBlock())
        .add(Linear.builder()
            .setUnits(10)
            .build())

    val lr = 0.9f
    val model = Model.newInstance("cnn")
    model.block = block

    var loss = Loss.softmaxCrossEntropyLoss()
    var lrt = Tracker.fixed(lr)
    var sgd = Optimizer.sgd().setLearningRateTracker(lrt).build()

    val config = DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
        .optDevices(Device.getDevices(1)) // Single GPU
        .addEvaluator(Accuracy()) // Model Accuracy
        .addTrainingListeners(*TrainingListener.Defaults.basic())

    val trainer = model.newTrainer(config)

    val X = manager.randomUniform(0f, 1.0f, Shape(1, 1, 28, 28))
    trainer.initialize(X.shape)

    var currentShape = X.shape

    for (i in 0 until block.children.size()) {
        val newShape = block.children[i].value.getOutputShapes(arrayOf(currentShape))
        currentShape = newShape[0]
        println(block.children[i].key.toString() + " layer output : " + currentShape)
    }

    val batchSize = 256
    val numEpochs = Integer.getInteger("MAX_EPOCH", 5)
    var trainLoss: DoubleArray
    var testAccuracy: DoubleArray
    val epochCount: DoubleArray
    var trainAccuracy: DoubleArray

    epochCount = DoubleArray(numEpochs)

    for (i in epochCount.indices) {
        epochCount[i] = (i + 1).toDouble()
    }

    val trainIter = FashionMnist.builder()
        .optUsage(Dataset.Usage.TRAIN)
        .setSampling(batchSize, true)
        .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
        .build()


    val testIter = FashionMnist.builder()
        .optUsage(Dataset.Usage.TEST)
        .setSampling(batchSize, true)
        .optLimit(java.lang.Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
        .build()

    trainIter.prepare()
    testIter.prepare()

    val temp = trainingChapter6(trainIter, testIter, numEpochs, trainer);
    trainLoss = temp[0]
    trainAccuracy = temp[1]
    testAccuracy = temp[2]

    val plot = Plot2DPanel()
    plot.addLinePlot("train Loss", Color.GREEN,epochCount,trainLoss)
    plot.addLinePlot("train Acc",Color.ORANGE,epochCount,trainAccuracy)
    plot.addLinePlot("test Acc",Color.BLUE,epochCount,testAccuracy)
    plot.addLegend("EAST")
    plot.setAxisLabel(0,"metrics")
    plot.setAxisLabel(1,"Epoch")
    val frame = JFrame("LeNet")
    frame.setSize(600, 600)
    frame.contentPane = plot
    frame.isVisible = true

}